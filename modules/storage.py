"""Persistent storage and small I/O helpers used by the transfer plugin."""

import asyncio
import json
import re
import time
from collections import OrderedDict
from pathlib import Path

import aiohttp
from astrbot.api import logger


def _sync_read_json(path: Path) -> dict:
    """Read JSON synchronously; callers run this in a worker thread."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error("文件不存在，本次创建空 JSON")
        return {}
    except json.JSONDecodeError as exc:
        logger.error(f"文件 {path} 不是有效 JSON: {exc}")
        raise ValueError(f"文件 {path} 不是有效 JSON: {exc}") from exc
    except OSError as exc:
        logger.error(f"读取文件 {path} 失败: {exc}")
        raise RuntimeError(f"读取文件 {path} 失败: {exc}") from exc
    except Exception as exc:
        logger.error(f"发生预期外的 JSON 读取错误: {exc}", exc_info=True)
        raise RuntimeError(f"发生预期外的 JSON 读取错误: {exc}") from exc


def _sync_write_json(path: Path, data: dict):
    """Write JSON atomically; callers run this in a worker thread."""
    temporary_path = path.with_suffix(".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        temporary_path.replace(path)
    except OSError as exc:
        logger.error(f"写入文件 {path} 失败: {exc}")
        raise RuntimeError(f"写入文件 {path} 失败: {exc}") from exc
    except TypeError as exc:
        logger.error(f"数据无法序列化为 JSON: {exc}")
        raise ValueError(f"数据无法序列化为 JSON: {exc}") from exc
    except Exception as exc:
        logger.error(f"发生预期外的 JSON 写入错误: {exc}")
        raise RuntimeError(f"发生预期外的 JSON 写入错误: {exc}") from exc
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


async def async_read_json(path: Path) -> dict:
    """Read JSON without blocking the asyncio event loop."""
    return await asyncio.to_thread(_sync_read_json, path)


async def async_write_json(path: Path, data: dict):
    """Write JSON without blocking the asyncio event loop."""
    await asyncio.to_thread(_sync_write_json, path, data)


def _classify_error(error: Exception) -> str:
    """Convert common transport errors to a short user-facing description."""
    if isinstance(error, asyncio.TimeoutError):
        return "请求超时，请稍后重试"
    if isinstance(error, aiohttp.ClientError):
        return "网络请求失败，请检查网络连接"
    if isinstance(error, (PermissionError, ConnectionRefusedError)):
        return "权限不足或连接被拒绝"
    if isinstance(error, (ValueError, KeyError, TypeError)):
        return str(error)
    return f"{error}"


class MsgTransferStore:
    """Persistent storage with one async lock and cache per data file."""

    MAX_MSG_MAPPINGS = 2000
    MSG_MAPPING_TRIM = 100
    MAX_FORWARD_LOG = 200
    FORWARD_LOG_TRIM = 50

    def __init__(
        self,
        webhook_file: Path,
        mapping_file: Path,
        msg_mapping_file: Path,
        forward_log_file: Path,
    ):
        self.webhook_file = webhook_file
        self.mapping_file = mapping_file
        self.msg_mapping_file = msg_mapping_file
        self.forward_log_file = forward_log_file
        self._ensure_files()

        self._webhook_lock = asyncio.Lock()
        self._mapping_lock = asyncio.Lock()
        self._msg_mapping_lock = asyncio.Lock()
        self._forward_log_lock = asyncio.Lock()

        self._webhooks = None
        self._mappings = None
        self._msg_mapping = None
        self._forward_log = None

        self._reverse_idx = None
        self._forward_text_idx = None

    def _ensure_files(self):
        self.webhook_file.parent.mkdir(parents=True, exist_ok=True)
        for file_path in (
            self.webhook_file,
            self.mapping_file,
            self.msg_mapping_file,
            self.forward_log_file,
        ):
            if not file_path.exists():
                file_path.write_text("{}", encoding="utf-8")

    async def _read_json(self, path: Path) -> dict:
        return await async_read_json(path)

    async def _write_json(self, path: Path, data: dict):
        await async_write_json(path, data)

    @staticmethod
    def _fuzzy_match_rule(source_umo: str, rules: dict) -> dict:
        fuzzy_matches = {}
        try:
            parts = source_umo.split(":")
            if len(parts) >= 3:
                platform = parts[0]
                message_type = parts[1]
                current_id = parts[2]
                for rule_id, rule in rules.items():
                    rule_source = rule["source_umo"]
                    rule_parts = rule_source.split(":")
                    if len(rule_parts) >= 3:
                        rule_platform, rule_type, rule_id_part = rule_parts[:3]
                        if rule_platform == platform and rule_type == message_type:
                            if len(rule_id_part) < 2 or len(current_id) < 2:
                                continue
                            if (
                                rule_id_part == current_id
                                or rule_id_part.endswith("_" + current_id)
                                or current_id.endswith("_" + rule_id_part)
                            ):
                                fuzzy_matches[rule_id] = rule
        except (KeyError, TypeError, ValueError, OSError) as exc:
            logger.error(f"[FuzzyMatch] 模糊匹配异常: {exc}")
        return fuzzy_matches

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
        """Return a read-only snapshot of the QQ name mapping."""
        async with self._mapping_lock:
            return await self._load_mappings()

    def _rebuild_reverse_idx(self):
        self._reverse_idx = {}
        if self._msg_mapping is None:
            return
        for qq_id, value in self._msg_mapping.items():
            if isinstance(value, dict):
                discord_msg_id = str(value.get("discord_msg_id", ""))
            else:
                discord_msg_id = (
                    value.split("|")[0]
                    if isinstance(value, str) and "|" in value
                    else str(value)
                )
            if discord_msg_id:
                self._reverse_idx[discord_msg_id] = qq_id

    async def _load_msg_mapping_raw(self) -> OrderedDict:
        """Load the mutable message mapping cache while holding its lock."""
        if self._msg_mapping is None:
            raw = await self._read_json(self.msg_mapping_file)
            self._msg_mapping = OrderedDict(raw)
            self._rebuild_reverse_idx()
        return self._msg_mapping

    async def _save_msg_mapping(self, data: OrderedDict):
        self._msg_mapping = data
        self._rebuild_reverse_idx()
        await self._write_json(self.msg_mapping_file, dict(data))

    async def set_msg_mapping(
        self,
        qq_msg_id: str,
        discord_msg_id: str,
        qq_user_id: str = "",
        qq_user_name: str = "",
        origin: str = "qq",
        forwarded_content: str = "",
    ):
        async with self._msg_mapping_lock:
            data = await self._load_msg_mapping_raw()

            if qq_msg_id in data:
                old_value = data[qq_msg_id]
                if isinstance(old_value, dict):
                    old_discord_msg_id = str(old_value.get("discord_msg_id", ""))
                else:
                    old_discord_msg_id = (
                        old_value.split("|")[0]
                        if isinstance(old_value, str) and "|" in old_value
                        else str(old_value)
                    )
                self._reverse_idx.pop(old_discord_msg_id, None)

            data[qq_msg_id] = {
                "discord_msg_id": str(discord_msg_id),
                "user_id": str(qq_user_id or ""),
                "user_name": str(qq_user_name or qq_user_id or ""),
                "origin": str(origin or "qq"),
                "forwarded_content": str(forwarded_content or ""),
            }

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
            value = data.get(qq_msg_id)
            if value is None:
                return None
            data.move_to_end(qq_msg_id)
            if isinstance(value, dict):
                discord_msg_id = value.get("discord_msg_id")
                return str(discord_msg_id) if discord_msg_id else None
            if isinstance(value, str) and "|" in value:
                return value.split("|")[0]
            return value

    async def get_msg_meta(self, qq_msg_id: str) -> dict | None:
        async with self._msg_mapping_lock:
            data = await self._load_msg_mapping_raw()
            value = data.get(qq_msg_id)
            if value is None:
                return None
            data.move_to_end(qq_msg_id)
            if isinstance(value, dict):
                return {
                    "user_id": str(value.get("user_id", "")),
                    "user_name": str(value.get("user_name") or value.get("user_id", "")),
                    "origin": str(value.get("origin") or "qq"),
                    "forwarded_content": str(value.get("forwarded_content") or ""),
                }
            if isinstance(value, str) and "|" in value:
                parts = value.split("|")
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

    def _rebuild_forward_idx(self):
        self._forward_text_idx = {}
        if self._forward_log is None:
            return
        for discord_msg_id, entry in self._forward_log.items():
            content = entry.get("content", "")
            timestamp = entry.get("timestamp", 0)
            sender_id = entry.get("sender_id", "")
            if content:
                existing = self._forward_text_idx.get(content)
                if existing is None or timestamp > existing[1]:
                    self._forward_text_idx[content] = (
                        discord_msg_id,
                        timestamp,
                        sender_id,
                    )

    @staticmethod
    def _normalize_forward_content(content: str) -> str:
        """Normalize forwarded text so QQ quotes can match Discord originals."""
        if not content:
            return ""

        normalized = str(content).replace("\u200b", "").replace("\ufeff", "")
        normalized = normalized.strip().replace("：", ":")
        forward_match = re.match(
            r"^\[转发\]\s+.+?(?:\s+\([^)]+\))?\s*:\s*(.*)$",
            normalized,
            flags=re.S,
        )
        if forward_match:
            normalized = forward_match.group(1).strip()
        return re.sub(r"\s+", " ", normalized)

    async def _find_forward_log_by_normalized_content(
        self,
        content: str,
    ) -> tuple[str, str | None] | None:
        """Find the newest Discord record by normalized content."""
        normalized = self._normalize_forward_content(content)
        if not normalized:
            return None

        async with self._forward_log_lock:
            data = await self._load_forward_log_raw()
            best = None
            for discord_msg_id, entry in data.items():
                entry_content = entry.get("content", "") if isinstance(entry, dict) else ""
                if self._normalize_forward_content(entry_content) != normalized:
                    continue
                timestamp = entry.get("timestamp", 0) if isinstance(entry, dict) else 0
                sender_id = entry.get("sender_id") if isinstance(entry, dict) else None
                if best is None or timestamp > best[2]:
                    best = (discord_msg_id, sender_id, timestamp)
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

    async def add_forward_log(
        self,
        discord_msg_id: str,
        content: str,
        sender_id: str = "",
    ):
        async with self._forward_log_lock:
            data = await self._load_forward_log_raw()
            timestamp = time.time()
            data[discord_msg_id] = {
                "content": content,
                "sender_id": sender_id,
                "timestamp": timestamp,
            }
            if len(data) > self.MAX_FORWARD_LOG:
                for key in sorted(data, key=lambda item: data[item]["timestamp"])[
                    : self.FORWARD_LOG_TRIM
                ]:
                    del data[key]
                self._rebuild_forward_idx()
            elif content:
                existing = self._forward_text_idx.get(content)
                if existing is None or timestamp > existing[1]:
                    self._forward_text_idx[content] = (
                        discord_msg_id,
                        timestamp,
                        sender_id,
                    )
            await self._save_forward_log(data)

    async def get_forward_entry_sender(self, discord_msg_id: str) -> str | None:
        async with self._forward_log_lock:
            data = await self._load_forward_log_raw()
            entry = data.get(discord_msg_id)
            return entry.get("sender_id") if entry else None

    async def find_forward_log_by_content(self, content: str) -> str | None:
        if not content:
            return None
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


__all__ = [
    "MsgTransferStore",
    "async_read_json",
    "async_write_json",
    "_classify_error",
]
