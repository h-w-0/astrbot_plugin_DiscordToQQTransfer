import asyncio
import json
import re
import secrets
import time
import urllib.parse
from collections import OrderedDict
from pathlib import Path

import aiohttp

import astrbot.api.star as star
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api import logger

import string

try:
    from astrbot.api.message_components import Plain, Reply, At, MessageChain
except ImportError:
    from astrbot.core.message.components import Plain, Reply, At
    from astrbot.core.message.message_event_result import MessageChain
from .webhook import DiscordWebhookManager

try:
    from astrbot.core.platform.astr_message_event import MessageSesion
except ImportError:
    MessageSesion = None


# ------------------------
# 工具与数据路径
# ------------------------


def _sync_read_json(path: Path) -> dict:
    """同步读 JSON（在 asyncio.to_thread 中执行）"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("❌ 文件不存在！本次创建空 JSON！")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"❌ 文件 {path} 不是有效 JSON: {e}")
        raise ValueError(f"❌ 文件 {path} 不是有效 JSON: {e}") from e
    except OSError as e:
        logger.error(f"❌ 读取文件 {path} 失败: {e}")
        raise RuntimeError(f"❌ 读取文件 {path} 失败: {e}") from e
    except Exception as e:
        logger.error(f"❌ 发生预期外的 JSON 读取错误: {e}", exc_info=True)
        raise RuntimeError(f"❌ 发生预期外的 JSON 读取错误: {e}")


def _sync_write_json(path: Path, data: dict):
    """同步写 JSON（在 asyncio.to_thread 中执行）"""
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except OSError as e:
        logger.error(f"❌ 写入文件 {path} 失败: {e}")
        raise RuntimeError(f"❌ 写入文件 {path} 失败: {e}") from e
    except TypeError as e:
        logger.error(f"❌ 数据无法序列化为 JSON: {e}")
        raise ValueError(f"❌ 数据无法序列化为 JSON: {e}") from e
    except Exception as e:
        logger.error(f"❌ 发生预期外的 JSON 写入错误: {e}")
        raise RuntimeError(f"❌ 发生预期外的 JSON 写入错误: {e}") from e
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


async def async_read_json(path: Path) -> dict:
    """异步读 JSON，通过线程池避免阻塞事件循环"""
    return await asyncio.to_thread(_sync_read_json, path)


async def async_write_json(path: Path, data: dict):
    """异步写 JSON（原子替换），通过线程池避免阻塞事件循环"""
    await asyncio.to_thread(_sync_write_json, path, data)


def gen_code(n=6):
    # 使用 secrets 模块生成更安全的随机字符串
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(n))


def _classify_error(e: Exception) -> str:
    """将异常分类为用户友好的描述"""
    if isinstance(e, asyncio.TimeoutError):
        return "请求超时，请稍后重试"
    if isinstance(e, aiohttp.ClientError):
        return "网络请求失败，请检查网络连接"
    if isinstance(e, (PermissionError, ConnectionRefusedError)):
        return "权限不足或连接被拒绝"
    if isinstance(e, (ValueError, KeyError, TypeError)):
        return str(e)
    return f"{e}"


# ------------------------
# 存储层（带异步锁与 LRU 淘汰）
# ------------------------
class MsgTransferStore:
    """持久化存储层 —— 每类数据独立 asyncio.Lock + 异步 I/O"""

    # 类常量
    MAX_MSG_MAPPINGS = 2000
    MSG_MAPPING_TRIM = 100
    MAX_FORWARD_LOG = 200
    FORWARD_LOG_TRIM = 50

    def __init__(self, rule_file: Path, pending_file: Path, webhook_file: Path,
                 mapping_file: Path, msg_mapping_file: Path, forward_log_file: Path):
        self.rule_file = rule_file
        self.pending_file = pending_file
        self.webhook_file = webhook_file
        self.mapping_file = mapping_file
        self.msg_mapping_file = msg_mapping_file
        self.forward_log_file = forward_log_file
        self._ensure_files()

        # ---- Per-file locks ----
        self._rule_lock = asyncio.Lock()
        self._pending_lock = asyncio.Lock()
        self._webhook_lock = asyncio.Lock()
        self._mapping_lock = asyncio.Lock()
        self._msg_mapping_lock = asyncio.Lock()
        self._forward_log_lock = asyncio.Lock()

        # ---- In-memory caches ----
        self._rules = None
        self._pending = None
        self._webhooks = None
        self._mappings = None
        self._msg_mapping = None
        self._forward_log = None

        # ---- Indexes ----
        self._reverse_idx = None       # discord_msg_id → qq_msg_id
        self._forward_text_idx = None   # content → (d_msg_id, ts, sender_id)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _ensure_files(self):
        self.rule_file.parent.mkdir(parents=True, exist_ok=True)
        for f in (self.rule_file, self.pending_file, self.webhook_file,
                  self.mapping_file, self.msg_mapping_file, self.forward_log_file):
            if not f.exists():
                f.write_text("{}", encoding="utf-8")

    async def _read_json(self, path: Path) -> dict:
        return await async_read_json(path)

    async def _write_json(self, path: Path, data: dict):
        await async_write_json(path, data)

    # ------------------------------------------------------------------ #
    # Rules
    # ------------------------------------------------------------------ #

    async def _load_rules(self) -> dict:
        if self._rules is None:
            self._rules = await self._read_json(self.rule_file)
        return dict(self._rules)

    async def _save_rules(self, data: dict):
        self._rules = data
        await self._write_json(self.rule_file, data)

    async def add_rule(self, source_umo: str, target_umo: str) -> str:
        async with self._rule_lock:
            data = await self._load_rules()
            for rid, rule in data.items():
                if rule["source_umo"] == source_umo and rule["target_umo"] == target_umo:
                    raise ValueError(f"规则已存在 #{rid}")
            new_id = str(max(map(int, data.keys()), default=0) + 1)
            data[new_id] = {"source_umo": source_umo, "target_umo": target_umo}
            await self._save_rules(data)
            return new_id

    async def delete_rule(self, rid: str):
        async with self._rule_lock:
            data = await self._load_rules()
            if rid not in data:
                raise KeyError("规则不存在")
            data.pop(rid)
            await self._save_rules(data)

    @staticmethod
    def _fuzzy_match_rule(source_umo: str, rules: dict) -> dict:
        fuzzy_matches = {}
        try:
            parts = source_umo.split(":")
            if len(parts) >= 3:
                platform = parts[0]
                msg_type = parts[1]
                current_id_part = parts[2]
                for rid, rule in rules.items():
                    rule_source = rule["source_umo"]
                    rule_parts = rule_source.split(":")
                    if len(rule_parts) >= 3:
                        r_platform, r_type, r_id = rule_parts[0], rule_parts[1], rule_parts[2]
                        if r_platform == platform and r_type == msg_type:
                            if len(r_id) < 2 or len(current_id_part) < 2:
                                continue
                            if (r_id == current_id_part
                                    or r_id.endswith("_" + current_id_part)
                                    or current_id_part.endswith("_" + r_id)):
                                fuzzy_matches[rid] = rule
        except (KeyError, TypeError, ValueError, OSError) as e:
            logger.error(f"[FuzzyMatch] 模糊匹配异常: {e}")
        return fuzzy_matches

    async def list_rules(self, source_umo):
        async with self._rule_lock:
            data = await self._load_rules()
        exact_matches = {rid: r for rid, r in data.items() if r["source_umo"] == source_umo}
        if exact_matches:
            return exact_matches
        return self._fuzzy_match_rule(source_umo, data)

    # ------------------------------------------------------------------ #
    # Pending
    # ------------------------------------------------------------------ #

    async def _load_pending(self) -> dict:
        if self._pending is None:
            self._pending = await self._read_json(self.pending_file)
        return dict(self._pending)

    async def _save_pending(self, data: dict):
        self._pending = data
        await self._write_json(self.pending_file, data)

    async def add_pending(self, code: str, source_umo: str):
        async with self._pending_lock:
            p = await self._load_pending()
            p[code] = {"source_umo": source_umo, "created_at": time.time()}
            await self._save_pending(p)

    async def pop_pending(self, code: str):
        async with self._pending_lock:
            p = await self._load_pending()
            entry = p.pop(code, None) if isinstance(p.get(code), dict) else p.pop(code, None)
            if entry is None:
                raise KeyError("绑定码不存在或已使用")
            await self._save_pending(p)
            return entry["source_umo"] if isinstance(entry, dict) else entry

    async def _cleanup_expired_pending(self, max_age: float = 86400):
        """清理超过 max_age 秒的待绑定请求"""
        async with self._pending_lock:
            p = await self._load_pending()
            now = time.time()
            changed = False
            for code, entry in list(p.items()):
                ts = entry.get("created_at", 0) if isinstance(entry, dict) else 0
                if ts and now - ts > max_age:
                    p.pop(code, None)
                    changed = True
            if changed:
                await self._save_pending(p)

    # ------------------------------------------------------------------ #
    # Webhooks
    # ------------------------------------------------------------------ #

    async def _load_webhooks(self) -> dict:
        if self._webhooks is None:
            self._webhooks = await self._read_json(self.webhook_file)
        return dict(self._webhooks)

    async def _save_webhooks(self, data: dict):
        self._webhooks = data
        await self._write_json(self.webhook_file, data)

    async def set_webhook_url(self, target_umo: str, webhook_url: str):
        async with self._webhook_lock:
            data = await self._load_webhooks()
            data[target_umo] = webhook_url
            await self._save_webhooks(data)

    async def get_webhook_url(self, target_umo: str) -> str | None:
        async with self._webhook_lock:
            data = await self._load_webhooks()
            return data.get(target_umo)

    async def remove_webhook_url(self, target_umo: str):
        async with self._webhook_lock:
            data = await self._load_webhooks()
            data.pop(target_umo, None)
            await self._save_webhooks(data)

    # ------------------------------------------------------------------ #
    # QQ number → name mapping
    # ------------------------------------------------------------------ #

    async def _load_mappings(self) -> dict:
        if self._mappings is None:
            self._mappings = await self._read_json(self.mapping_file)
        return dict(self._mappings)

    async def _save_mappings(self, data: dict):
        self._mappings = data
        await self._write_json(self.mapping_file, data)

    async def update_mapping(self, qq_id: str, qq_name: str) -> bool:
        async with self._mapping_lock:
            data = await self._load_mappings()
            if data.get(qq_id) != qq_name:
                data[qq_id] = qq_name
                await self._save_mappings(data)
                return True
            return False

    async def load_mappings(self):
        """公开只读接口 —— 返回浅拷贝"""
        async with self._mapping_lock:
            return await self._load_mappings()

    # ------------------------------------------------------------------ #
    # msg_id mapping (QQ msg_id ↔ Discord msg_id)
    # ------------------------------------------------------------------ #

    def _rebuild_reverse_idx(self):
        self._reverse_idx = {}
        if self._msg_mapping is None:
            return
        for qq_id, val in self._msg_mapping.items():
            d_id = val.split('|')[0] if isinstance(val, str) and '|' in val else str(val)
            self._reverse_idx[d_id] = qq_id

    async def _load_msg_mapping_raw(self) -> OrderedDict:
        """加载原始 msg_mapping 缓存（可变的，在锁内通过 move_to_end 追踪 LRU）"""
        if self._msg_mapping is None:
            raw = await self._read_json(self.msg_mapping_file)
            self._msg_mapping = OrderedDict(raw)
            self._rebuild_reverse_idx()
        return self._msg_mapping

    async def _save_msg_mapping(self, data: OrderedDict):
        self._msg_mapping = data
        self._rebuild_reverse_idx()
        await self._write_json(self.msg_mapping_file, dict(data))

    async def set_msg_mapping(self, qq_msg_id: str, discord_msg_id: str,
                              qq_user_id: str = "", qq_user_name: str = "", origin: str = "qq"):
        async with self._msg_mapping_lock:
            data = await self._load_msg_mapping_raw()

            if qq_msg_id in data:
                old_val = data[qq_msg_id]
                old_d_id = old_val.split('|')[0] if isinstance(old_val, str) and '|' in old_val else str(old_val)
                self._reverse_idx.pop(old_d_id, None)

            if qq_user_id:
                if origin and origin != "qq":
                    data[qq_msg_id] = f"{discord_msg_id}|{qq_user_id}|{qq_user_name}|{origin}"
                else:
                    data[qq_msg_id] = f"{discord_msg_id}|{qq_user_id}|{qq_user_name}"
            else:
                data[qq_msg_id] = discord_msg_id

            if len(data) > self.MAX_MSG_MAPPINGS:
                for _ in range(self.MSG_MAPPING_TRIM):
                    try:
                        data.popitem(last=False)
                    except KeyError:
                        break
                self._rebuild_reverse_idx()
            else:
                self._reverse_idx[str(discord_msg_id)] = qq_msg_id

            await self._save_msg_mapping(data)

    async def get_msg_mapping(self, qq_msg_id: str) -> str | None:
        async with self._msg_mapping_lock:
            data = await self._load_msg_mapping_raw()
            val = data.get(qq_msg_id)
            if val is None:
                return None
            data.move_to_end(qq_msg_id)
            if isinstance(val, str) and '|' in val:
                return val.split('|')[0]
            return val

    async def get_msg_meta(self, qq_msg_id: str) -> dict | None:
        async with self._msg_mapping_lock:
            data = await self._load_msg_mapping_raw()
            val = data.get(qq_msg_id)
            if val is None:
                return None
            data.move_to_end(qq_msg_id)
            if isinstance(val, str) and '|' in val:
                parts = val.split('|')
                return {
                    "user_id": parts[1],
                    "user_name": parts[2] if len(parts) > 2 else parts[1],
                    "origin": parts[3] if len(parts) > 3 else "qq",
                }
            return None

    async def find_qq_msg_id_by_discord_id(self, discord_msg_id: str) -> str | None:
        async with self._msg_mapping_lock:
            if self._reverse_idx is None:
                await self._load_msg_mapping_raw()
            return self._reverse_idx.get(str(discord_msg_id))

    # ------------------------------------------------------------------ #
    # Forward log (Discord→QQ 消息记录)
    # ------------------------------------------------------------------ #

    def _rebuild_forward_idx(self):
        self._forward_text_idx = {}
        if self._forward_log is None:
            return
        for d_msg_id, entry in self._forward_log.items():
            content = entry.get("content", "")
            ts = entry.get("timestamp", 0)
            sid = entry.get("sender_id", "")
            if content:
                existing = self._forward_text_idx.get(content)
                if existing is None or ts > existing[1]:
                    self._forward_text_idx[content] = (d_msg_id, ts, sid)

    @staticmethod
    def _normalize_forward_content(content: str) -> str:
        """规范化转发文本，降低 QQ 引用文本与 Discord 原文的格式差异影响"""
        if not content:
            return ""

        normalized = str(content).replace("\u200b", "").replace("\ufeff", "")
        normalized = normalized.strip().replace("：", ":")
        fwd_match = re.match(r"^\[转发\]\s+.+?(?:\s+\([^)]+\))?\s*:\s*(.*)$", normalized, flags=re.S)
        if fwd_match:
            normalized = fwd_match.group(1).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    async def _find_forward_log_by_normalized_content(self, content: str) -> tuple[str, str | None] | None:
        """按规范化文本查找最新的 Discord 转发记录，返回 (discord_msg_id, sender_id)"""
        normalized = self._normalize_forward_content(content)
        if not normalized:
            return None

        async with self._forward_log_lock:
            data = await self._load_forward_log_raw()
            best = None
            for d_msg_id, entry in data.items():
                entry_content = entry.get("content", "") if isinstance(entry, dict) else ""
                if self._normalize_forward_content(entry_content) != normalized:
                    continue
                ts = entry.get("timestamp", 0) if isinstance(entry, dict) else 0
                sender_id = entry.get("sender_id") if isinstance(entry, dict) else None
                if best is None or ts > best[2]:
                    best = (d_msg_id, sender_id, ts)
            if best:
                return best[0], best[1]
            return None

    async def _load_forward_log_raw(self) -> dict:
        if self._forward_log is None:
            self._forward_log = await self._read_json(self.forward_log_file)
            self._rebuild_forward_idx()
        return self._forward_log

    async def _save_forward_log(self, data: dict):
        self._forward_log = data
        self._rebuild_forward_idx()
        await self._write_json(self.forward_log_file, data)

    async def add_forward_log(self, discord_msg_id: str, content: str, sender_id: str = ""):
        async with self._forward_log_lock:
            data = await self._load_forward_log_raw()
            ts = time.time()
            data[discord_msg_id] = {"content": content, "sender_id": sender_id, "timestamp": ts}
            if len(data) > self.MAX_FORWARD_LOG:
                for k in sorted(data, key=lambda k: data[k]["timestamp"])[:self.FORWARD_LOG_TRIM]:
                    del data[k]
                self._rebuild_forward_idx()
            elif content:
                existing = self._forward_text_idx.get(content)
                if existing is None or ts > existing[1]:
                    self._forward_text_idx[content] = (discord_msg_id, ts, sender_id)
            await self._save_forward_log(data)

    async def get_forward_entry_sender(self, discord_msg_id: str) -> str | None:
        async with self._forward_log_lock:
            data = await self._load_forward_log_raw()
            entry = data.get(discord_msg_id)
            return entry.get("sender_id") if entry else None

    async def find_forward_log_by_content(self, content: str) -> str | None:
        if not content:
            return None
        # 文本索引是同步快照，无需锁
        if self._forward_text_idx is None:
            async with self._forward_log_lock:
                await self._load_forward_log_raw()
        result = self._forward_text_idx.get(content)
        if result:
            return result[0]
        normalized_result = await self._find_forward_log_by_normalized_content(content)
        return normalized_result[0] if normalized_result else None

    async def find_forward_log_sender(self, content: str) -> str | None:
        if not content:
            return None
        if self._forward_text_idx is None:
            async with self._forward_log_lock:
                await self._load_forward_log_raw()
        result = self._forward_text_idx.get(content)
        if result and len(result) > 2:
            return result[2]
        normalized_result = await self._find_forward_log_by_normalized_content(content)
        return normalized_result[1] if normalized_result else None


# ------------------------
# 插件主体
# ------------------------
class MsgTransfer(star.Star):
    def __init__(self, context: Context):
        super().__init__(context)
        # 使用 AstrBot 提供的标准方法获取项目持久化数据存储目录
        self.data_dir = star.StarTools.get_data_dir("astrbot_plugin_DiscordToQQTransfer")
        self.rule_file = self.data_dir / "rules.json"
        self.forward_log_file = self.data_dir / "forward_log.json"
        self.pending_file = self.data_dir / "pending.json"
        self.webhook_file = self.data_dir / "webhooks.json"
        self.mapping_file = self.data_dir / "mappings.json"
        self.msg_mapping_file = self.data_dir / "msg_mapping.json"

        self.store = MsgTransferStore(self.rule_file, self.pending_file, self.webhook_file, self.mapping_file, self.msg_mapping_file, self.forward_log_file)
        self.webhook_manager = DiscordWebhookManager(context)

    async def initialize(self):
        """预缓存 Discord 客户端，避免每次转发时重复扫描 star_map"""
        client = self.webhook_manager.get_discord_client()
        if client:
            logger.info("MsgTransfer: Discord 客户端已缓存")
        else:
            logger.info("MsgTransfer: Discord 客户端未就绪（非 Discord 环境或 py-cord 未安装）")
        await self.store._cleanup_expired_pending()
        logger.info("MsgTransfer plugin init OK")

    @filter.command_group("mt")
    def mt(self, event: AstrMessageEvent):
        """mt 命令组"""
        pass

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mt.command("add")
    async def cmd_add(self, event: AstrMessageEvent):
        """创建一则消息转发绑定的请求"""
        code = gen_code()
        source_umo = str(event.unified_msg_origin)
        await self.store.add_pending(code, source_umo)

        yield event.plain_result(
            f"📌 已创建绑定请求\n"
            f"请在目标会话执行：#mt bind {code}"
        )

    @mt.command("bind")
    async def cmd_bind(self, event: AstrMessageEvent, code: str):
        """接受一则消息转发绑定的请求"""
        try:
            target_umo = str(event.unified_msg_origin)
            source_umo = await self.store.pop_pending(code)
            rid = await self.store.add_rule(source_umo, target_umo)

            # 如果目标是Discord，自动创建Webhook
            target_platform = event.get_platform_name()
            is_discord = target_platform == "discord" or "discord" in target_umo.lower()
            webhook_ok = False

            if is_discord:
                channel_id = None
                parts = target_umo.split(":")
                if len(parts) >= 3:
                    channel_id = parts[2]
                elif len(parts) == 2:
                    channel_id = parts[1]

                if channel_id:
                    webhook_url = await self.webhook_manager.create_webhook_for_channel(int(channel_id))
                    if webhook_url:
                        await self.store.set_webhook_url(target_umo, webhook_url)
                        webhook_ok = True

            if is_discord and not webhook_ok:
                yield event.plain_result(
                    f"✅ 绑定成功 #{rid}，但自动创建 Webhook 失败，请检查机器人权限或手动设置 Webhook URL"
                )
            else:
                yield event.plain_result(f"✅ 绑定成功 #{rid}")
        except (ValueError, KeyError) as e:
            yield event.plain_result(f"❌ 绑定失败：{e}")
        except PermissionError:
            logger.error("[Bind] 权限不足")
            yield event.plain_result("❌ 绑定失败：权限不足，请检查机器人权限设置")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            logger.error("[Bind] 网络请求失败")
            yield event.plain_result("❌ 绑定失败：网络请求超时或连接失败，请稍后重试")
        except Exception as e:
            logger.error(f"[Bind] 绑定异常: {e}", exc_info=True)
            yield event.plain_result(f"❌ 绑定失败：{_classify_error(e)}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @mt.command("del")
    async def cmd_del(self, event: AstrMessageEvent, rid: str):
        """删除一条转发规则"""
        try:
            await self.store.delete_rule(rid)
            yield event.plain_result(f"🗑️ 已删除规则 #{rid}")
        except (KeyError, ValueError, OSError, RuntimeError) as e:
            yield event.plain_result(f"❌ 删除失败: {e}")

    @mt.command("list")
    async def cmd_list(self, event: AstrMessageEvent):
        """列出与当前会话相关的所有转发规则"""
        source_umo = str(event.unified_msg_origin)
        rules = await self.store.list_rules(source_umo)
        if not rules:
            yield event.plain_result("📭 当前没有转发规则")
            return

        lines = [f"📜 转发规则（{len(rules)}条）"]
        for rid, r in rules.items():
            lines.append(f"#{rid}")
        yield event.plain_result("\n".join(lines))

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def forward_message(self, event: AstrMessageEvent):
        """主转发逻辑 - 顺序队列处理所有转发规则，保证顺序一致"""
        try:
            source_umo = str(event.unified_msg_origin)
            rules = await self.store.list_rules(source_umo)
            if not rules:
                return
            message_chain = event.get_messages()
            # 记录从 Discord 转发的消息，供 QQ 回复引用时还原跳转链接
            platform = event.get_platform_name()
            if platform == "discord":
                discord_msg_id = event.message_obj.message_id
                if discord_msg_id:
                    msg_text = DiscordWebhookManager.format_message_content(message_chain)
                    if msg_text:
                        await self.store.add_forward_log(str(discord_msg_id), msg_text, event.get_sender_id())
            # 顺序依次await每个转发，保证顺序
            for rid, rule in rules.items():
                await self._forward_single_rule(event, rule, rid, source_umo, message_chain)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"❌ 转发逻辑异常: {e}", exc_info=True)

    async def _forward_single_rule(self, event: AstrMessageEvent, rule: dict, rid: str, source_umo: str, message_chain):
        """处理单个转发规则"""
        try:
            # 自动记录QQ号和名称到 mapping_file
            platform = event.get_platform_name()
            if platform in ["aiocqhttp", "qqofficial"]:
                qq_id = event.get_sender_id()
                qq_name = event.get_sender_name()
                if await self.store.update_mapping(qq_id, qq_name):
                    logger.info(f"转发时已更新QQ号 {qq_id} 的名称: {qq_name}")

            target = rule["target_umo"]
            webhook_url = await self.store.get_webhook_url(target)
            if webhook_url:
                await self._forward_with_webhook(event, target, message_chain, rid, webhook_url)
                return

            # 非 webhook 目标（如 QQ），通过 AstrBot 框架发送
            try:
                sender_name = event.get_sender_name()
                source_platform_name = event.get_platform_name()
                msg_text = DiscordWebhookManager.format_message_content(message_chain)
                if msg_text:
                    full_text = f"[转发] {sender_name} ({source_platform_name})​: {msg_text}"
                else:
                    full_text = f"[转发] {sender_name} ({source_platform_name})​"

                # Discord 端回复消息时，检测引用关系并还原 QQ 引用链
                chain = await self._build_discord_reply_chain(event, source_platform_name, sender_name, msg_text, full_text)
                sent, sent_result = await self._send_message_with_result(target, chain)
                if sent:
                    await self._record_discord_to_target_mapping(event, sent_result, source_platform_name)
                    logger.info(f"已转发 #{rid} -> {target}")
                else:
                    logger.warning(f"转发 #{rid} 未找到目标平台适配器: {target}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"通过 AstrBot 转发 #{rid} 网络错误: {e}")
            except (OSError, ValueError, KeyError) as e:
                logger.error(f"通过 AstrBot 转发 #{rid} 失败: {e}")
        except (KeyError, ValueError, OSError, RuntimeError) as e:
            logger.error(f"❌ 处理规则 #{rid} 时发生异常: {e}")

    async def _build_discord_reply_chain(self, event, source_platform_name, sender_name, msg_text, full_text):
        """构建非 webhook 转发的消息链，还原 Discord 回复引用关系"""
        chain_parts = []
        if source_platform_name == "discord":
            _raw = getattr(event.message_obj, 'raw_message', None)
            if _raw:
                _ref = getattr(_raw, 'reference', None)
                if _ref and _ref.message_id:
                    orig_qq_id = await self.store.find_qq_msg_id_by_discord_id(str(_ref.message_id))
                    if orig_qq_id:
                        meta = await self.store.get_msg_meta(orig_qq_id)
                        if meta and meta.get("origin", "qq") == "qq":
                            chain_parts.append(Reply(id=orig_qq_id))
                            chain_parts.append(Plain(text=f"[转发] {sender_name} ({source_platform_name}):"))
                            chain_parts.append(At(qq=meta['user_id']))
                            if msg_text:
                                chain_parts.append(Plain(text=f" {msg_text}"))
                        else:
                            chain_parts.append(Reply(id=orig_qq_id))

        if not chain_parts:
            chain_parts.append(Plain(text=full_text))
        elif not any(isinstance(c, At) for c in chain_parts):
            chain_parts.append(Plain(text=full_text))
        chain = MessageChain()
        chain.chain = chain_parts
        return chain

    @staticmethod
    def _extract_message_id_from_send_result(result) -> str | None:
        """从不同平台 send_by_session 返回值中尽力提取发送后的消息 ID"""
        if result is None or isinstance(result, bool):
            return None

        if isinstance(result, (str, int)):
            return str(result)

        if isinstance(result, dict):
            for key in ("message_id", "msg_id", "id"):
                value = result.get(key)
                if value:
                    return str(value)
            for key in ("data", "result", "message"):
                nested = result.get(key)
                nested_id = MsgTransfer._extract_message_id_from_send_result(nested)
                if nested_id:
                    return nested_id
            return None

        for attr in ("message_id", "msg_id", "id"):
            value = getattr(result, attr, None)
            if value:
                return str(value)

        if isinstance(result, (list, tuple)):
            for item in result:
                nested_id = MsgTransfer._extract_message_id_from_send_result(item)
                if nested_id:
                    return nested_id

        return None

    async def _send_message_with_result(self, target: str, chain: MessageChain) -> tuple[bool, object | None]:
        """发送消息并保留底层平台返回值；失败时保持 Context.send_message 的语义"""
        if MessageSesion is None:
            sent = await self.context.send_message(target, chain)
            return bool(sent), sent

        session = MessageSesion.from_str(target) if isinstance(target, str) else target
        platform_manager = getattr(self.context, "platform_manager", None)
        platform_insts = getattr(platform_manager, "platform_insts", []) if platform_manager else []
        for platform in platform_insts:
            meta = platform.meta()
            if meta.name != session.platform_name:
                continue
            result = await platform.send_by_session(session, chain)
            return True, result

        sent = await self.context.send_message(target, chain)
        return bool(sent), sent

    async def _record_discord_to_target_mapping(self, event: AstrMessageEvent, sent_result, source_platform_name: str):
        """记录 Discord→QQ 转发后目标平台消息 ID 到 Discord 原消息 ID 的映射"""
        if source_platform_name != "discord":
            return

        discord_msg_id = getattr(event.message_obj, "message_id", None)
        if not discord_msg_id:
            return

        sent_msg_id = self._extract_message_id_from_send_result(sent_result)
        if not sent_msg_id:
            logger.debug("Discord→QQ 转发成功，但无法从发送结果提取目标消息 ID，保留文本回退匹配")
            return

        try:
            await self.store.set_msg_mapping(
                str(sent_msg_id),
                str(discord_msg_id),
                event.get_sender_id(),
                event.get_sender_name(),
                origin="discord",
            )
        except Exception as e:
            logger.error(f"保存 Discord→QQ 消息映射失败(不影响发送): {e}")

    # ---- Webhook 辅助方法 ----

    @staticmethod
    def _extract_quote_info(message_chain):
        """从消息链中提取引用/回复信息"""
        quote_text = None
        quote_sender = None
        reply_to_qq_id = None
        for seg in message_chain:
            if seg.__class__.__name__ in ("Quote", "Reply"):
                if hasattr(seg, "origin_text"):
                    quote_text = seg.origin_text
                if hasattr(seg, "origin_sender"):
                    quote_sender = seg.origin_sender
                if hasattr(seg, "text") and not quote_text:
                    quote_text = seg.text
                if hasattr(seg, "sender_name") and not quote_sender:
                    quote_sender = seg.sender_name
                if hasattr(seg, "sender_nickname") and not quote_sender:
                    quote_sender = seg.sender_nickname
                if not quote_text and hasattr(seg, "message_str") and seg.message_str:
                    quote_text = seg.message_str
                if not quote_text and hasattr(seg, "chain") and seg.chain:
                    for sub in seg.chain:
                        if sub.__class__.__name__ == "File":
                            fname = getattr(sub, "name", None) or getattr(sub, "filename", None) or "文件"
                            quote_text = f"[{fname}]"
                            break
                if hasattr(seg, "id") and seg.id:
                    reply_to_qq_id = str(seg.id)
                break
        return quote_text, quote_sender, reply_to_qq_id

    @staticmethod
    def _resolve_forward_quote(quote_text, quote_sender):
        """解析 [转发] 前缀，返回 (quote_text, quote_sender, discord_sender_name)"""
        discord_sender_name = None
        if quote_text and quote_text.strip().startswith('[转发]'):
            fwd_match = re.match(
                r"^\[转发\]\s+(.+?)(?:\s+\([^)]+\))?​?\s*[：:]\s*(.*)",
                quote_text.strip()
            )
            if fwd_match:
                parsed_sender = fwd_match.group(1).strip()
                parsed_text = fwd_match.group(2).strip()
                if parsed_sender:
                    quote_sender = parsed_sender
                if parsed_text:
                    parsed_text = re.sub(r'@[^\s(]+\(\d+\)\s*', '', parsed_text[:500]).strip()
                    if parsed_text:
                        quote_text = parsed_text
                if parsed_sender:
                    discord_sender_name = parsed_sender
        return quote_text, quote_sender, discord_sender_name

    async def _resolve_reply_target(self, reply_to_qq_id, quote_text, target_umo):
        """解析回复目标，返回 (reply_to_discord_id, discord_sender_id, jump_url)"""
        reply_to_discord_id = None
        discord_sender_id = None
        if reply_to_qq_id:
            reply_to_discord_id = await self.store.get_msg_mapping(reply_to_qq_id)
            if reply_to_discord_id:
                meta = await self.store.get_msg_meta(reply_to_qq_id)
                if meta and meta.get("origin") == "discord":
                    discord_sender_id = meta.get("user_id")

        if reply_to_discord_id is None and quote_text:
            fwd_discord_id = await self.store.find_forward_log_by_content(quote_text)
            if fwd_discord_id:
                reply_to_discord_id = fwd_discord_id
                discord_sender_id = await self.store.get_forward_entry_sender(fwd_discord_id)

        jump_url = None
        if reply_to_discord_id:
            channel_id = None
            parts = target_umo.split(":")
            if len(parts) >= 3:
                try:
                    channel_id = int(parts[2])
                except (ValueError, TypeError):
                    channel_id = None
            if channel_id:
                try:
                    client = self.webhook_manager.get_discord_client()
                    if client:
                        channel = await client.fetch_channel(channel_id)
                        if hasattr(channel, 'guild') and channel.guild:
                            guild_id = channel.guild.id
                            jump_url = f"https://discord.com/channels/{guild_id}/{channel_id}/{reply_to_discord_id}"
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError) as e:
                    logger.warning(f"构建 Discord jump URL 失败: {e}")

        return reply_to_discord_id, discord_sender_id, jump_url

    @staticmethod
    def _replace_ats(message_chain, discord_sender_id, discord_sender_name, mapping, self_id):
        """将 At(QQ) 替换为 Discord 兼容的提及格式"""
        new_chain = []
        for seg in message_chain:
            if seg.__class__.__name__ == "At" and hasattr(seg, "qq"):
                qq_id = str(seg.qq)
                if self_id and qq_id == self_id:
                    if discord_sender_id:
                        new_chain.append(Plain(text=f"<@{discord_sender_id}> "))
                    elif discord_sender_name:
                        new_chain.append(Plain(text=f"@{discord_sender_name} "))
                else:
                    qq_name = mapping.get(qq_id, qq_id)
                    new_chain.append(Plain(text=f"@{qq_name} "))
            elif seg.__class__.__name__ in ("Quote", "Reply"):
                continue
            else:
                new_chain.append(seg)
        return new_chain

    @staticmethod
    def _build_webhook_quote(content, reply_to_discord_id, jump_url, quote_text, quote_sender):
        """为 webhook 消息添加引用块"""
        if reply_to_discord_id:
            prefix = f"**{quote_sender}**: " if quote_sender else ""
            if jump_url:
                label = quote_text or "引用消息"
                return f"> {prefix}[{label}]({jump_url})\n{content}"
            elif quote_text:
                return f"> {prefix}{quote_text}\n{content}"
            return content

        if quote_text:
            prefix = f"**{quote_sender}**: " if quote_sender else ""
            _is_img = False
            if quote_text.startswith(('http://', 'https://')):
                _path = urllib.parse.urlparse(quote_text).path.lower()
                _is_img = _path.endswith(('.jpg', '.png', '.jpeg', '.gif', '.webp'))
            quote_block = f"> {prefix}[图片]({quote_text})\n" if _is_img else f"> {prefix}{quote_text}\n"
            return quote_block + content

        return content

    async def _forward_with_webhook(self, event: AstrMessageEvent, target_umo: str, message_chain, rule_id: str, webhook_url: str) -> bool:
        try:
            sender_name = event.get_sender_name()
            sender_id = event.get_sender_id()
            source_platform = event.get_platform_name()
            self_id = event.get_self_id()
            mapping = await self.store.load_mappings()

            # Step 1-2: 提取并解析引用信息
            quote_text, quote_sender, reply_to_qq_id = self._extract_quote_info(message_chain)
            quote_text, quote_sender, discord_sender_name = self._resolve_forward_quote(quote_text, quote_sender)

            # Step 3: 解析 Discord 端回复目标
            reply_to_discord_id, discord_sender_id, jump_url = await self._resolve_reply_target(
                reply_to_qq_id, quote_text, target_umo
            )

            # Step 4: 替换 @提及
            new_chain = self._replace_ats(message_chain, discord_sender_id, discord_sender_name, mapping, self_id)

            # Step 5-6: 构建 webhook 内容（含引用块）和 embeds
            virtual_username = DiscordWebhookManager.build_virtual_username(sender_name, source_platform)
            avatar_url = DiscordWebhookManager.get_avatar_url(source_platform, sender_id)
            # 提取图片并转为 Discord embeds，内容中不再包含图片 URL
            image_urls = DiscordWebhookManager.extract_images(new_chain)
            raw_content = DiscordWebhookManager.format_message_content(new_chain, skip_images=True)
            content = self._build_webhook_quote(raw_content, reply_to_discord_id, jump_url, quote_text, quote_sender)
            embeds = [{"image": {"url": url}} for url in image_urls[:10]]  # Discord 最多 10 个 embed

            # Step 7: 发送并记录映射
            discord_msg_id = await self.webhook_manager.send_webhook_message(
                webhook_url=webhook_url,
                username=virtual_username,
                avatar_url=avatar_url,
                content=content,
                embeds=embeds if embeds else None,
            )

            if discord_msg_id:
                qq_msg_id = event.message_obj.message_id
                if qq_msg_id:
                    qq_user_id = event.get_sender_id()
                    qq_user_name = event.get_sender_name()
                    try:
                        await self.store.set_msg_mapping(qq_msg_id, discord_msg_id, qq_user_id, qq_user_name)
                    except Exception as e:
                        logger.error(f"保存消息映射 #{rule_id} 失败(不影响发送): {e}")
                return True

            return False
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"❌ Webhook网络错误 #{rule_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Webhook转发异常 #{rule_id}: {_classify_error(e)}")
            return False

    async def terminate(self):
        await self.webhook_manager.close()
        logger.info("MsgTransfer plugin terminated")
