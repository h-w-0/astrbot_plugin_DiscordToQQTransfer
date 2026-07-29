"""Message-chain normalization and Discord-facing formatting helpers."""

import re
import urllib.parse

from astrbot.api import logger

try:
    from astrbot.api.message_components import Plain
except ImportError:
    from astrbot.core.message.components import Plain

from ..webhook import DiscordWebhookManager
from .translation import _TRANSLATION_LITERAL_RE


class MessageProcessingMixin:
    """Pure message transformations shared by all forwarding paths."""

    @staticmethod
    def _component_kind(component) -> str:
        """Return a stable component type name across AstrBot versions and fakes."""
        raw_type = None
        if isinstance(component, dict):
            raw_type = component.get("type")
            if raw_type is None and isinstance(component.get("data"), dict):
                raw_type = component["data"].get("type")
        else:
            raw_type = getattr(component, "type", None)

        if hasattr(raw_type, "value"):
            raw_type = raw_type.value
        if raw_type:
            kind = str(raw_type).split(".")[-1].strip().lower()
            if kind == "text":
                return "plain"
            return kind
        if hasattr(component, "nodes"):
            return "nodes"
        if hasattr(component, "content") and (
            component.__class__.__name__.strip().lower() == "node"
            or hasattr(component, "name")
            or hasattr(component, "uin")
        ):
            return "node"
        if hasattr(component, "text"):
            return "plain"
        return component.__class__.__name__.strip().lower()

    @classmethod
    def _is_node_component(cls, component) -> bool:
        return cls._component_kind(component) in {"node", "nodes"}

    @classmethod
    def _node_children(cls, component) -> list:
        """Extract Node/Nodes children, returning an empty list for malformed data."""
        kind = cls._component_kind(component)
        if kind == "node":
            return [component]
        if kind != "nodes":
            return []

        if isinstance(component, dict):
            payload = component.get("data") if isinstance(component.get("data"), dict) else component
            children = payload.get("nodes") or payload.get("messages") or []
        else:
            children = getattr(component, "nodes", [])

        if isinstance(children, dict):
            children = [children]
        if isinstance(children, (list, tuple)):
            return list(children)
        return []

    @staticmethod
    def _node_content(node) -> list:
        if isinstance(node, dict):
            payload = node.get("data") if isinstance(node.get("data"), dict) else node
            content = payload.get("content") or payload.get("message") or []
        else:
            content = getattr(node, "content", [])

        if content is None:
            return []
        if isinstance(content, dict):
            return [content]
        if isinstance(content, (list, tuple)):
            return list(content)
        return [content]

    @staticmethod
    def _node_sender(node) -> str:
        if isinstance(node, dict):
            payload = node.get("data") if isinstance(node.get("data"), dict) else node
            name = payload.get("nickname") or payload.get("name")
            uin = payload.get("user_id") or payload.get("uin")
        else:
            name = getattr(node, "name", None)
            uin = getattr(node, "uin", None)

        name = re.sub(r"[\r\n\t]+", " ", str(name or "")).strip()
        uin = str(uin or "").strip()
        return name or uin or "未知用户"

    @staticmethod
    def _merged_forward_header(unit: dict) -> str:
        path = unit.get("path") or ()
        path_text = ".".join(str(value) for value in path) or "附加"
        depth = max(0, int(unit.get("depth", 0) or 0))
        label = "嵌套转发" if depth else "转发记录"
        if unit.get("continuation"):
            path_text += " 续"
        indent = "↳ " * depth
        return f"{indent}【{label} {path_text}】 {unit.get('sender') or '未知用户'} (QQ)"

    def _append_forward_node_units(
        self,
        node,
        path: tuple[int, ...],
        depth: int,
        units: list[dict],
    ) -> None:
        sender = self._node_sender(node)
        current_components = []
        emitted = False
        continuation = False
        nested_index = 0

        def flush_current() -> None:
            nonlocal current_components, emitted
            if current_components or not emitted:
                units.append({
                    "path": path,
                    "depth": depth,
                    "sender": sender,
                    "components": list(current_components),
                    "continuation": continuation and emitted,
                })
                emitted = True
            current_components = []

        for component in self._node_content(node):
            if self._is_node_component(component):
                flush_current()
                nested_nodes = self._node_children(component)
                if not nested_nodes:
                    nested_index += 1
                    logger.warning(
                        f"[MergedForward] 无法解析嵌套转发节点，路径={'.'.join(str(value) for value in path + (nested_index,))}"
                    )
                    units.append({
                        "path": path + (nested_index,),
                        "depth": depth + 1,
                        "sender": "未知用户",
                        "components": [],
                        "continuation": False,
                        "placeholder": "[无法解析的嵌套转发节点]",
                    })
                else:
                    for child in nested_nodes:
                        nested_index += 1
                        self._append_forward_node_units(
                            child,
                            path + (nested_index,),
                            depth + 1,
                            units,
                        )
                continuation = True
            else:
                current_components.append(component)

        flush_current()

    def _build_merged_forward_units(self, message_chain) -> list[dict]:
        """Expand merged-forward nodes in source order while retaining hierarchy."""
        if message_chain is None:
            components = []
        elif isinstance(message_chain, (list, tuple)):
            components = list(message_chain)
        else:
            components = [message_chain]

        units: list[dict] = []
        ordinary_components = []
        root_index = 0

        def flush_ordinary() -> None:
            if not ordinary_components:
                return
            units.append({
                "path": (),
                "depth": 0,
                "sender": "附加内容",
                "components": list(ordinary_components),
                "continuation": False,
            })
            ordinary_components.clear()

        for component in components:
            if not self._is_node_component(component):
                ordinary_components.append(component)
                continue

            flush_ordinary()
            nested_nodes = self._node_children(component)
            if not nested_nodes:
                logger.warning("[MergedForward] 合并转发节点为空或格式异常")
                units.append({
                    "path": (),
                    "depth": 0,
                    "sender": "未知用户",
                    "components": [],
                    "continuation": False,
                    "placeholder": "[空合并转发]",
                })
                continue

            for node in nested_nodes:
                root_index += 1
                self._append_forward_node_units(node, (root_index,), 0, units)

        flush_ordinary()
        if not units:
            units.append({
                "path": (),
                "depth": 0,
                "sender": "未知用户",
                "components": [],
                "continuation": False,
                "placeholder": "[空合并转发]",
            })
        return units

    @classmethod
    def _is_merged_forward_message(cls, message_chain) -> bool:
        if message_chain is None:
            return False
        components = message_chain if isinstance(message_chain, (list, tuple)) else [message_chain]
        return any(cls._component_kind(component) in {"node", "nodes"} for component in components)

    def _format_merged_forward_text(self, message_chain) -> str:
        """Extract text from merged-forward content for safety review."""
        text_lines = []
        for unit in self._build_merged_forward_units(message_chain):
            values = []
            for component in unit.get("components", []):
                kind = self._component_kind(component)
                if kind == "plain":
                    if isinstance(component, dict):
                        payload = component.get("data") if isinstance(component.get("data"), dict) else component
                        value = payload.get("text")
                    else:
                        value = getattr(component, "text", None)
                    if value:
                        values.append(str(value))
                elif kind == "at":
                    qq = (
                        component.get("data", {}).get("qq")
                        if isinstance(component, dict)
                        else getattr(component, "qq", "")
                    )
                    values.append(f"@{qq}" if qq else "[艾特]")
            value = "".join(values).strip()
            if value:
                text_lines.append(f"{self._merged_forward_header(unit)}: {value}")
            elif unit.get("placeholder"):
                text_lines.append(
                    f"{self._merged_forward_header(unit)}: {unit['placeholder']}"
                )
        return "\n".join(text_lines)

    def _format_merged_component_content(self, components) -> str:
        """Format merged-forward components and keep unsupported types visible."""
        renderable_components = []
        for component in components:
            if not isinstance(component, dict):
                renderable_components.append(component)
                continue

            payload = component.get("data") if isinstance(component.get("data"), dict) else component
            kind = self._component_kind(component)
            if kind == "plain":
                renderable_components.append(Plain(text=str(payload.get("text") or "")))
            elif kind == "at":
                qq = payload.get("qq") or payload.get("user_id")
                renderable_components.append(Plain(text=f"<@{qq}>" if qq else "[艾特]"))
            elif kind == "image":
                source = payload.get("url") or payload.get("file") or payload.get("path")
                renderable_components.append(Plain(text=str(source) if source else "[图片]"))
            elif kind == "file":
                name = payload.get("name") or "文件"
                source = payload.get("url") or payload.get("file") or ""
                renderable_components.append(
                    Plain(text=f"[{name}]({source})" if source else f"[{name}]")
                )
            else:
                renderable_components.append(component)

        rendered = DiscordWebhookManager.format_message_content(
            renderable_components,
            skip_images=True,
        )
        extra_lines = []
        handled = {"plain", "at", "image", "file"}
        for component in components:
            kind = self._component_kind(component)
            if kind in handled:
                continue
            if kind in {"reply", "quote"}:
                if isinstance(component, dict):
                    payload = component.get("data") if isinstance(component.get("data"), dict) else component
                    quote_text = payload.get("message_str") or payload.get("text") or payload.get("origin_text")
                else:
                    quote_text = (
                        getattr(component, "origin_text", None)
                        or getattr(component, "message_str", None)
                        or getattr(component, "text", None)
                    )
                extra_lines.append(f"[引用] {quote_text or '原消息'}")
                continue
            if kind in {"record", "audio"}:
                source = (
                    getattr(component, "url", None)
                    or getattr(component, "file", None)
                    or getattr(component, "path", None)
                )
                extra_lines.append(
                    f"[语音]({source})"
                    if isinstance(source, str) and source.startswith("http")
                    else "[语音]"
                )
                continue
            if kind == "video":
                source = (
                    getattr(component, "url", None)
                    or getattr(component, "file", None)
                    or getattr(component, "path", None)
                )
                extra_lines.append(
                    f"[视频]({source})"
                    if isinstance(source, str) and source.startswith("http")
                    else "[视频]"
                )
                continue

            logger.warning(f"[MergedForward] 不支持的消息组件类型: {component.__class__.__name__}")
            extra_lines.append(f"[不支持的消息类型: {component.__class__.__name__}]")

        if extra_lines:
            if rendered and not rendered.endswith("\n"):
                rendered += "\n"
            rendered += "\n".join(extra_lines)
        return rendered

    def _is_forward_record_translation_enabled(self, rule: dict | None) -> bool:
        """Translate merged-forward record text only when explicitly enabled."""
        if not self._is_translation_enabled_for_rule(rule):
            return False
        translation = rule.get("translation", {}) if isinstance(rule, dict) else {}
        if not isinstance(translation, dict):
            return False
        return self._coerce_config_bool(translation.get("translate_forward_records"), False)

    async def _translate_forward_record_components(
        self,
        event,
        components: list,
        rule: dict,
    ) -> list:
        if not self._is_forward_record_translation_enabled(rule):
            return list(components)

        translated_components = []
        text_buffer: list[str] = []

        async def flush_text_buffer() -> None:
            if not text_buffer:
                return
            original_text = "".join(text_buffer)
            text_buffer.clear()
            stripped_text = original_text.strip()
            if not stripped_text or _TRANSLATION_LITERAL_RE.fullmatch(stripped_text):
                translated_components.append(Plain(text=original_text))
                return
            translated_text = None
            try:
                translated_text = await self._translate_message(event, original_text, rule)
            except Exception as exc:
                logger.warning(f"[MergedForward] 转发记录文本翻译失败，回退原文: {exc}")
            translated_components.append(Plain(text=translated_text or original_text))

        for component in components:
            if self._component_kind(component) == "plain":
                value = (
                    component.get("data", {}).get("text")
                    if isinstance(component, dict)
                    else getattr(component, "text", "")
                )
                text_buffer.append(str(value or ""))
            else:
                await flush_text_buffer()
                translated_components.append(component)
        await flush_text_buffer()
        return translated_components

    @staticmethod
    def _split_discord_content(content: str, max_length: int = 2000) -> list[str]:
        """Split text at Discord's limit, preferring newline boundaries."""
        content = str(content or "")
        if len(content) <= max_length:
            return [content]

        chunks = []
        remaining = content
        while remaining:
            if len(remaining) <= max_length:
                chunks.append(remaining)
                break
            split_at = remaining.rfind("\n", 0, max_length)
            if split_at > 0:
                chunks.append(remaining[: split_at + 1])
                remaining = remaining[split_at + 1 :]
                continue
            chunks.append(remaining[:max_length])
            remaining = remaining[max_length:]
        return chunks

    @staticmethod
    def _discord_channel_id(target_umo: str) -> int | None:
        parts = str(target_umo or "").split(":")
        channel_id = parts[2] if len(parts) >= 3 else parts[1] if len(parts) == 2 else ""
        try:
            return int(channel_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _thread_id(thread) -> str | None:
        if isinstance(thread, dict):
            value = thread.get("id") or thread.get("thread_id")
        else:
            value = getattr(thread, "id", None) or getattr(thread, "thread_id", None)
        return str(value) if value is not None else None

    @staticmethod
    def _event_value(event, method_name: str, default: str = "") -> str:
        getter = getattr(event, method_name, None)
        if callable(getter):
            try:
                return str(getter() or default)
            except Exception:
                return default
        return default

    async def _prepare_merged_forward_unit(
        self,
        event,
        unit: dict,
        rule: dict,
        mapping: dict,
    ) -> tuple[str, list, list[str]]:
        protected_mentions: dict[str, str] = {}
        components = self._replace_ats(
            unit.get("components", []),
            discord_sender_id=None,
            discord_sender_name=None,
            mapping=mapping,
            self_id=self._event_value(event, "get_self_id"),
            protected_mentions=protected_mentions,
            drop_replies=False,
        )
        components = await self._translate_forward_record_components(event, components, rule)
        raw_content = self._format_merged_component_content(components)
        raw_content = self._restore_translation_literals(raw_content, protected_mentions)
        placeholder = unit.get("placeholder")
        if placeholder:
            raw_content = f"{placeholder}\n{raw_content}" if raw_content else placeholder

        header = self._merged_forward_header(unit)
        content = f"{header}\n{raw_content}" if raw_content else header
        image_urls = DiscordWebhookManager.extract_images(components)
        local_images = DiscordWebhookManager.extract_local_image_paths(components)
        embeds = [{"image": {"url": url}} for url in image_urls[:10]]
        return content, embeds, local_images[:10]

    @staticmethod
    def _extract_quote_info(message_chain):
        """Extract quote text, sender and QQ reply ID from a message chain."""
        quote_text = None
        quote_sender = None
        reply_to_qq_id = None
        for segment in message_chain:
            if segment.__class__.__name__ in ("Quote", "Reply"):
                if hasattr(segment, "origin_text"):
                    quote_text = segment.origin_text
                if hasattr(segment, "origin_sender"):
                    quote_sender = segment.origin_sender
                if hasattr(segment, "text") and not quote_text:
                    quote_text = segment.text
                if hasattr(segment, "sender_name") and not quote_sender:
                    quote_sender = segment.sender_name
                if hasattr(segment, "sender_nickname") and not quote_sender:
                    quote_sender = segment.sender_nickname
                if not quote_text and hasattr(segment, "message_str") and segment.message_str:
                    quote_text = segment.message_str
                if not quote_text and hasattr(segment, "chain") and segment.chain:
                    for sub_segment in segment.chain:
                        if sub_segment.__class__.__name__ == "File":
                            file_name = (
                                getattr(sub_segment, "name", None)
                                or getattr(sub_segment, "filename", None)
                                or "文件"
                            )
                            quote_text = f"[{file_name}]"
                            break
                if hasattr(segment, "id") and segment.id:
                    reply_to_qq_id = str(segment.id)
                break
        return quote_text, quote_sender, reply_to_qq_id

    @staticmethod
    def _resolve_forward_quote(quote_text, quote_sender):
        """Parse a [转发] prefix into quote text, sender and Discord sender name."""
        discord_sender_name = None
        if quote_text and quote_text.strip().startswith("[转发]"):
            forward_match = re.match(
                r"^\[转发\]\s+(.+?)(?:\s+\([^)]+\))?​?\s*[：:]\s*(.*)",
                quote_text.strip(),
            )
            if forward_match:
                parsed_sender = forward_match.group(1).strip()
                parsed_text = forward_match.group(2).strip()
                if parsed_sender:
                    quote_sender = parsed_sender
                if parsed_text:
                    parsed_text = re.sub(r"@[^\s(]+\(\d+\)\s*", "", parsed_text[:500]).strip()
                    if parsed_text:
                        quote_text = parsed_text
                if parsed_sender:
                    discord_sender_name = parsed_sender
        return quote_text, quote_sender, discord_sender_name

    @staticmethod
    def _replace_ats(
        message_chain,
        discord_sender_id,
        discord_sender_name,
        mapping,
        self_id,
        protected_mentions: dict[str, str] | None = None,
        drop_replies: bool = True,
    ):
        """Replace QQ At components with Discord-compatible mention text."""
        new_chain = []
        for segment in message_chain:
            if segment.__class__.__name__ == "At" and hasattr(segment, "qq"):
                qq_id = str(segment.qq)
                if self_id and qq_id == self_id:
                    if discord_sender_id:
                        mention_text = f"<@{discord_sender_id}> "
                    elif discord_sender_name:
                        mention_text = f"@{discord_sender_name} "
                    else:
                        mention_text = ""
                else:
                    qq_name = mapping.get(qq_id, qq_id)
                    mention_text = f"@{qq_name} "

                if mention_text:
                    if protected_mentions is not None:
                        token = f"__ASTRBOT_AT_{len(protected_mentions)}__"
                        protected_mentions[token] = mention_text
                        new_chain.append(Plain(text=token))
                    else:
                        new_chain.append(Plain(text=mention_text))
            elif drop_replies and segment.__class__.__name__ in ("Quote", "Reply"):
                continue
            else:
                new_chain.append(segment)
        return new_chain

    @staticmethod
    def _build_webhook_quote(content, reply_to_discord_id, jump_url, quote_text, quote_sender):
        """Add a Markdown quote block to a Discord webhook message."""
        if reply_to_discord_id:
            prefix = f"**{quote_sender}**: " if quote_sender else ""
            if jump_url:
                label = quote_text or "引用消息"
                return f"> {prefix}[{label}]({jump_url})\n{content}"
            if quote_text:
                return f"> {prefix}{quote_text}\n{content}"
            return content

        if quote_text:
            prefix = f"**{quote_sender}**: " if quote_sender else ""
            is_image = False
            if quote_text.startswith(("http://", "https://")):
                path = urllib.parse.urlparse(quote_text).path.lower()
                is_image = path.endswith((".jpg", ".png", ".jpeg", ".gif", ".webp"))
            quote_block = (
                f"> {prefix}[图片]({quote_text})\n"
                if is_image
                else f"> {prefix}{quote_text}\n"
            )
            return quote_block + content

        return content


__all__ = ["MessageProcessingMixin"]
