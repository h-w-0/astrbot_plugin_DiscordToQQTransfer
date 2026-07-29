"""Message-chain normalization and Discord-facing formatting helpers."""

import json
import re
import urllib.parse

from astrbot.api import logger

try:
    from astrbot.api.message_components import Plain
except ImportError:
    from astrbot.core.message.components import Plain

from ..webhook import DiscordWebhookManager
from .translation import _TRANSLATION_LITERAL_RE

try:
    from astrbot.core.utils.quoted_message.onebot_client import OneBotClient
except ImportError:
    OneBotClient = None


class MessageProcessingMixin:
    """Pure message transformations shared by all forwarding paths."""

    MERGED_FORWARD_CONTEXT_SIZE = 5

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

    @staticmethod
    def _forward_reference_id(component) -> str | None:
        """Return a OneBot forward ID from a Forward component or raw segment."""
        if isinstance(component, dict):
            payload = component.get("data") if isinstance(component.get("data"), dict) else component
            value = payload.get("id") or payload.get("message_id") or payload.get("forward_id")
        else:
            value = (
                getattr(component, "id", None)
                or getattr(component, "message_id", None)
                or getattr(component, "forward_id", None)
            )
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @classmethod
    def _is_forward_reference_component(cls, component) -> bool:
        return cls._component_kind(component) in {"forward", "forward_msg"}

    @classmethod
    def _contains_forward_reference(cls, message_chain) -> bool:
        """Find Forward(id) components, including ones nested in Node content."""
        if message_chain is None:
            return False
        components = message_chain if isinstance(message_chain, (list, tuple)) else [message_chain]
        seen: set[int] = set()

        def visit(component) -> bool:
            if cls._is_forward_reference_component(component):
                return True
            identity = id(component)
            if identity in seen:
                return False
            seen.add(identity)
            kind = cls._component_kind(component)
            if kind == "nodes":
                return any(visit(child) for child in cls._node_children(component))
            if kind == "node":
                return any(visit(child) for child in cls._node_content(component))
            return False

        return any(visit(component) for component in components)

    @staticmethod
    def _forward_placeholder_node(text: str) -> dict:
        return {
            "type": "node",
            "data": {
                "user_id": "",
                "nickname": "未知用户",
                "content": [],
            },
            "_merged_forward_placeholder": text,
        }

    @classmethod
    def _normalize_forward_content(cls, content) -> list:
        """Normalize OneBot node content without interpreting message semantics."""
        if content is None:
            return []
        if isinstance(content, str):
            text = content.strip()
            if not text:
                return []
            try:
                decoded = json.loads(text)
            except (TypeError, ValueError):
                decoded = None
            if isinstance(decoded, list):
                content = decoded
            else:
                return [{"type": "text", "data": {"text": content}}]
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, (list, tuple)):
            return [{"type": "text", "data": {"text": str(content)}}]

        normalized = []
        for component in content:
            if isinstance(component, str):
                normalized.append({"type": "text", "data": {"text": component}})
            else:
                normalized.append(component)
        return normalized

    @classmethod
    def _normalize_forward_node(cls, node) -> dict:
        """Convert a get_forward_msg node into the local Node-like dictionary form."""
        if not isinstance(node, dict):
            return cls._forward_placeholder_node(
                f"[不支持的合并转发节点: {node.__class__.__name__}]"
            )

        node_payload = node.get("data") if isinstance(node.get("data"), dict) else node
        sender = node.get("sender") if isinstance(node.get("sender"), dict) else None
        if sender is None and isinstance(node_payload.get("sender"), dict):
            sender = node_payload["sender"]
        sender = sender or {}

        nickname = (
            sender.get("nickname")
            or sender.get("card")
            or node_payload.get("nickname")
            or node_payload.get("name")
            or "未知用户"
        )
        user_id = (
            sender.get("user_id")
            or sender.get("uin")
            or node_payload.get("user_id")
            or node_payload.get("uin")
            or ""
        )
        if "content" in node_payload:
            content = node_payload.get("content")
        else:
            content = node_payload.get("message")
        return {
            "type": "node",
            "data": {
                "user_id": str(user_id or ""),
                "nickname": str(nickname or "未知用户"),
                "content": cls._normalize_forward_content(content),
            },
        }

    @classmethod
    def _extract_forward_nodes(cls, payload) -> list[dict]:
        """Extract the known OneBot get_forward_msg response shapes."""
        if isinstance(payload, list):
            raw_nodes = payload
            source_key = None
        elif isinstance(payload, dict):
            data = payload
            raw_nodes = None
            source_key = None
            for _ in range(3):
                for key in ("messages", "message", "nodes", "nodeList"):
                    if isinstance(data, dict) and key in data:
                        raw_nodes = data[key]
                        source_key = key
                        break
                if raw_nodes is not None:
                    break
                nested_data = data.get("data") if isinstance(data, dict) else None
                if not isinstance(nested_data, dict):
                    break
                data = nested_data
            if raw_nodes is None and isinstance(data, dict):
                if any(key in data for key in ("sender", "content", "message")):
                    raw_nodes = [data]
        else:
            raw_nodes = None
            source_key = None

        # ``get_msg`` also uses ``message`` for ordinary segments.  Do not
        # turn a text/image segment list into fake forwarded nodes.
        if source_key == "message" and isinstance(raw_nodes, (list, tuple)):
            if raw_nodes and not any(cls._looks_like_forward_node(node) for node in raw_nodes):
                raw_nodes = None
        if isinstance(raw_nodes, dict):
            raw_nodes = [raw_nodes]
        if not isinstance(raw_nodes, (list, tuple)):
            return []
        return [cls._normalize_forward_node(node) for node in raw_nodes]

    @staticmethod
    def _looks_like_forward_node(node) -> bool:
        if not isinstance(node, dict):
            return False
        node_type = str(node.get("type") or "").strip().lower()
        if node_type == "node":
            return True
        node_data = node.get("data") if isinstance(node.get("data"), dict) else node
        if isinstance(node.get("sender"), dict):
            return True
        return any(
            key in node_data
            for key in ("nickname", "name", "user_id", "uin", "content")
        ) and "message" in node_data

    @classmethod
    def _extract_single_forward_node(cls, payload) -> list[dict]:
        """Extract a single node returned by a get_msg-compatible endpoint."""
        data = payload
        for _ in range(3):
            if not isinstance(data, dict):
                return []
            if (
                isinstance(data.get("sender"), dict)
                and ("message" in data or "content" in data)
            ):
                return [cls._normalize_forward_node(data)]
            nested_data = data.get("data")
            if not isinstance(nested_data, dict):
                return []
            data = nested_data
        return []

    @classmethod
    def _extract_inline_forward_nodes(cls, component) -> list[dict]:
        """Extract nodes embedded in a Forward segment by some OneBot adapters."""
        if isinstance(component, dict):
            payload = component.get("data") if isinstance(component.get("data"), dict) else component
        else:
            payload = {
                key: getattr(component, key, None)
                for key in ("content", "messages", "nodes", "nodeList")
                if getattr(component, key, None) is not None
            }

        raw_nodes = None
        for key in ("messages", "nodes", "nodeList", "content"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if value:
                raw_nodes = value
                break
        if raw_nodes is None:
            return []
        if isinstance(raw_nodes, dict):
            raw_nodes = [raw_nodes]
        if not isinstance(raw_nodes, (list, tuple)):
            return []

        if raw_nodes and any(cls._looks_like_forward_node(node) for node in raw_nodes):
            return [cls._normalize_forward_node(node) for node in raw_nodes]
        # A few adapters expose one node's message chain as content directly.
        return [
            cls._normalize_forward_node(
                {
                    "sender": {},
                    "message": list(raw_nodes),
                }
            )
        ]

    async def _fetch_forward_payload(self, event, forward_id: str):
        """Fetch a forward record through AstrBot's OneBot client abstraction."""
        if OneBotClient is not None:
            return await OneBotClient(event).get_forward_msg(forward_id)

        # Older AstrBot versions may not expose OneBotClient yet.
        bot = getattr(event, "bot", None)
        api = getattr(bot, "api", None)
        call_action = getattr(api, "call_action", None)
        if not callable(call_action):
            return None
        params_list = [{"message_id": forward_id}, {"id": forward_id}]
        if forward_id.isdigit():
            numeric_id = int(forward_id)
            params_list.extend([{"message_id": numeric_id}, {"id": numeric_id}])
        for params in params_list:
            try:
                result = await call_action("get_forward_msg", **params)
            except Exception as exc:
                logger.debug(f"[MergedForward] 获取转发记录参数 {params} 失败: {exc}")
                continue
            if isinstance(result, dict):
                return result
        return None

    @staticmethod
    def _forward_fallback_ids(event, forward_id: str) -> list[str]:
        """Return outer event IDs used by adapters that expose inner IDs."""
        candidates = []
        message_obj = getattr(event, "message_obj", None)
        raw_message = getattr(message_obj, "raw_message", None)
        values = [getattr(message_obj, "message_id", None)]
        if isinstance(raw_message, dict):
            values.append(raw_message.get("message_id"))
        else:
            values.append(getattr(raw_message, "message_id", None))
        for value in values:
            if value is None:
                continue
            value = str(value).strip()
            if value and value != forward_id and value not in candidates:
                candidates.append(value)
        return candidates

    async def _fetch_message_payload(self, event, message_id: str):
        """Fetch one message as a fallback for adapters exposing inner nodes."""
        if OneBotClient is not None:
            return await OneBotClient(event).get_msg(message_id)

        bot = getattr(event, "bot", None)
        api = getattr(bot, "api", None)
        call_action = getattr(api, "call_action", None)
        if not callable(call_action):
            return None
        for params in ({"message_id": message_id}, {"id": message_id}):
            try:
                result = await call_action("get_msg", **params)
            except Exception as exc:
                logger.debug(f"[MergedForward] 获取单条转发节点 {params} 失败: {exc}")
                continue
            if isinstance(result, dict):
                return result
        return None

    async def _resolve_forward_nodes(
        self,
        event,
        nodes: list[dict],
        cache: dict[str, dict],
        active_ids: tuple[str, ...],
    ) -> dict:
        resolved_nodes = []
        for node in nodes:
            try:
                resolved_nodes.append(
                    await self._resolve_merged_forward_component(
                        event,
                        node,
                        cache,
                        active_ids,
                        allow_outer_fallback=False,
                    )
                )
            except Exception as exc:
                logger.warning(
                    f"[MergedForward] 单个转发节点解析失败，保留节点占位: {exc}"
                )
                resolved_nodes.append(
                    self._forward_placeholder_node("[单个转发节点解析失败]")
                )
        return {"type": "nodes", "data": {"nodes": resolved_nodes}}

    async def _resolve_merged_forward_component(
        self,
        event,
        component,
        cache: dict[str, dict],
        active_ids: tuple[str, ...] = (),
        allow_outer_fallback: bool = False,
    ):
        """Resolve one component while retaining Node/Nodes structure for rendering."""
        kind = self._component_kind(component)
        if self._is_forward_reference_component(component):
            forward_id = self._forward_reference_id(component)
            if not forward_id:
                logger.warning("[MergedForward] Forward 组件缺少记录 ID")
                return self._forward_placeholder_node("[合并转发解析失败: 缺少记录 ID]")
            if forward_id in active_ids:
                path = " -> ".join((*active_ids, forward_id))
                logger.warning(f"[MergedForward] 检测到循环嵌套转发，路径={path}")
                return self._forward_placeholder_node(
                    f"[合并转发循环引用: {forward_id}]"
                )
            if forward_id in cache:
                return cache[forward_id]

            nodes = self._extract_inline_forward_nodes(component)
            if nodes:
                logger.info(f"[MergedForward] 已使用 Forward 内联节点，id={forward_id}")
            else:
                try:
                    payload = await self._fetch_forward_payload(event, forward_id)
                except Exception as exc:
                    logger.warning(
                        f"[MergedForward] 获取合并转发失败，id={forward_id}: {exc}"
                    )
                    payload = None

                nodes = self._extract_forward_nodes(payload)
            outer_fallback_ids = (
                self._forward_fallback_ids(event, forward_id)
                if allow_outer_fallback
                else []
            )
            if not nodes:
                for fallback_id in outer_fallback_ids:
                    try:
                        fallback_payload = await self._fetch_forward_payload(
                            event,
                            fallback_id,
                        )
                        nodes = self._extract_forward_nodes(fallback_payload)
                    except Exception as exc:
                        logger.debug(
                            f"[MergedForward] 使用外层消息 ID {fallback_id} 获取转发失败: {exc}"
                        )
                        nodes = []
                    if nodes:
                        logger.info(
                            f"[MergedForward] Forward.id={forward_id} 无法直接获取，已使用外层消息 ID {fallback_id}"
                        )
                        break

            if not nodes:
                fallback_ids = [forward_id, *outer_fallback_ids]
                for fallback_id in fallback_ids:
                    try:
                        fallback_payload = await self._fetch_message_payload(
                            event,
                            fallback_id,
                        )
                        nodes = self._extract_single_forward_node(fallback_payload)
                    except Exception as exc:
                        logger.debug(
                            f"[MergedForward] 使用 get_msg 获取节点 {fallback_id} 失败: {exc}"
                        )
                        nodes = []
                    if nodes:
                        logger.info(
                            f"[MergedForward] 已通过 get_msg 恢复转发节点，id={fallback_id}"
                        )
                        break

            if not nodes:
                logger.warning(f"[MergedForward] 合并转发数据为空或格式异常，id={forward_id}")
                placeholder = self._forward_placeholder_node(
                    f"[合并转发解析失败: {forward_id}]"
                )
                cache[forward_id] = {"type": "nodes", "data": {"nodes": [placeholder]}}
                return cache[forward_id]

            resolved = await self._resolve_forward_nodes(
                event,
                nodes,
                cache,
                (*active_ids, forward_id),
            )
            cache[forward_id] = resolved
            return resolved

        if kind == "nodes":
            children = []
            for child in self._node_children(component):
                children.append(
                    await self._resolve_merged_forward_component(
                        event,
                        child,
                        cache,
                        active_ids,
                        allow_outer_fallback=False,
                    )
                )
            return {"type": "nodes", "data": {"nodes": children}}

        if kind == "node":
            payload = component.get("data") if isinstance(component, dict) and isinstance(component.get("data"), dict) else None
            content = self._node_content(component)
            resolved_content = []
            for child in content:
                resolved_content.append(
                    await self._resolve_merged_forward_component(
                        event,
                        child,
                        cache,
                        active_ids,
                        allow_outer_fallback=False,
                    )
                )
            if payload is None:
                return {
                    "type": "node",
                    "data": {
                        "user_id": str(getattr(component, "uin", "") or ""),
                        "nickname": str(getattr(component, "name", "") or "未知用户"),
                        "content": resolved_content,
                    },
                }
            resolved_node = dict(component)
            resolved_node["data"] = dict(payload)
            resolved_node["data"]["content"] = resolved_content
            return resolved_node

        return component

    async def _resolve_merged_forward_message(self, event, message_chain):
        """Resolve remote Forward(id) segments once before per-rule forwarding."""
        if not self._contains_forward_reference(message_chain):
            return message_chain
        components = message_chain if isinstance(message_chain, (list, tuple)) else [message_chain]
        cache: dict[str, dict] = {}
        resolved = []
        for component in components:
            try:
                resolved.append(
                    await self._resolve_merged_forward_component(
                        event,
                        component,
                        cache,
                        allow_outer_fallback=self._is_forward_reference_component(component),
                    )
                )
            except Exception as exc:
                logger.error(
                    f"[MergedForward] 解析消息组件失败，保留占位内容: {exc}",
                    exc_info=True,
                )
                resolved.append(component)
        return resolved

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
            sender = payload.get("sender") if isinstance(payload.get("sender"), dict) else {}
            name = (
                sender.get("nickname")
                or sender.get("card")
                or payload.get("nickname")
                or payload.get("name")
            )
            uin = sender.get("user_id") or sender.get("uin") or payload.get("user_id") or payload.get("uin")
        else:
            name = getattr(node, "name", None)
            uin = getattr(node, "uin", None)

        name = re.sub(r"[\r\n\t]+", " ", str(name or "")).strip()
        uin = str(uin or "").strip()
        return name or uin or "未知用户"

    def _merged_forward_header(self, unit: dict, target_language: str = "Chinese") -> str:
        path = unit.get("path") or ()
        path_text = ".".join(str(value) for value in path) or self._merged_forward_label(
            "additional", target_language
        )
        depth = max(0, int(unit.get("depth", 0) or 0))
        label = self._merged_forward_label(
            "nested_forward" if depth else "forward_record",
            target_language,
        )
        if unit.get("continuation"):
            path_text += f" {self._merged_forward_label('continued', target_language)}"
        indent = "↳ " * depth
        return f"{indent}【{label} {path_text}】 {unit.get('sender') or '未知用户'} (QQ)"

    def _merged_forward_label(self, key: str, target_language: str = "Chinese") -> str:
        labels = {
            "merged_forward": ("合并转发", "Merged Forward"),
            "forward_record": ("转发记录", "Forward Record"),
            "nested_forward": ("嵌套转发", "Nested Forward"),
            "additional": ("附加", "Additional"),
            "continued": ("续", "continued"),
            "mention": ("[艾特]", "[Mention]"),
            "emoji": ("[表情]", "[Emoji]"),
            "image": ("[图片]", "[Image]"),
            "file": ("文件", "File"),
            "quote": ("[引用]", "[Quote]"),
            "audio": ("[语音]", "[Audio]"),
            "video": ("[视频]", "[Video]"),
            "unsupported": ("不支持的消息类型", "Unsupported message type"),
        }
        chinese, english = labels.get(key, (key, key))
        return chinese if self._is_chinese_language(target_language) else english

    def _localize_merged_forward_placeholder(
        self,
        placeholder: str | None,
        target_language: str = "Chinese",
    ) -> str | None:
        if not placeholder or self._is_chinese_language(target_language):
            return placeholder
        localized = str(placeholder)
        replacements = (
            ("[空合并转发]", "[Empty merged forward]"),
            ("[无法解析的嵌套转发节点]", "[Unparseable nested forward node]"),
            ("[合并转发解析失败: 缺少记录 ID]", "[Merged forward parsing failed: missing record ID]"),
            ("[合并转发循环引用: ", "[Circular merged-forward reference: "),
            ("[合并转发解析失败: ", "[Merged forward parsing failed: "),
        )
        for source, translated in replacements:
            localized = localized.replace(source, translated)
        return localized

    def _append_forward_node_units(
        self,
        node,
        path: tuple[int, ...],
        depth: int,
        units: list[dict],
    ) -> None:
        sender = self._node_sender(node)
        node_placeholder = (
            node.get("_merged_forward_placeholder")
            if isinstance(node, dict)
            else None
        )
        current_components = []
        emitted = False
        continuation = False
        nested_index = 0

        def flush_current() -> None:
            nonlocal current_components, emitted
            if current_components or not emitted:
                unit = {
                    "path": path,
                    "depth": depth,
                    "sender": sender,
                    "components": list(current_components),
                    "continuation": continuation and emitted,
                }
                if node_placeholder:
                    unit["placeholder"] = node_placeholder
                units.append(unit)
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
            elif self._is_forward_reference_component(component):
                flush_current()
                nested_index += 1
                logger.warning(
                    f"[MergedForward] 发现未解析的嵌套转发，路径={'.'.join(str(value) for value in path + (nested_index,))}"
                )
                units.append({
                    "path": path + (nested_index,),
                    "depth": depth + 1,
                    "sender": "未知用户",
                    "components": [],
                    "continuation": False,
                    "placeholder": f"[合并转发解析失败: {self._forward_reference_id(component) or '未知 ID'}]",
                })
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
            if self._is_forward_reference_component(component):
                flush_ordinary()
                root_index += 1
                forward_id = self._forward_reference_id(component) or "未知 ID"
                logger.warning(f"[MergedForward] 发现未解析的合并转发，id={forward_id}")
                units.append({
                    "path": (root_index,),
                    "depth": 0,
                    "sender": "未知用户",
                    "components": [],
                    "continuation": False,
                    "placeholder": f"[合并转发解析失败: {forward_id}]",
                })
                continue
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
        return any(
            cls._component_kind(component) in {"node", "nodes", "forward", "forward_msg"}
            for component in components
        )

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

    @staticmethod
    def _merged_forward_context_group(unit: dict) -> tuple:
        """Return the parent node path so context stays within one nesting level."""
        path = unit.get("path") or ()
        if isinstance(path, str):
            path = (path,)
        return tuple(path[:-1])

    def _build_merged_forward_translation_context(
        self,
        units: list[dict],
        current_index: int,
        target_language: str = "Chinese",
    ) -> list[dict[str, str]]:
        """Build up to five preceding/following sibling records for translation."""
        if current_index < 0 or current_index >= len(units):
            return []

        current_group = self._merged_forward_context_group(units[current_index])
        sibling_indices = [
            index
            for index, unit in enumerate(units)
            if self._merged_forward_context_group(unit) == current_group
        ]
        try:
            sibling_position = sibling_indices.index(current_index)
        except ValueError:
            return []

        window_size = self.MERGED_FORWARD_CONTEXT_SIZE
        context_indices = (
            sibling_indices[max(0, sibling_position - window_size):sibling_position]
            + sibling_indices[sibling_position + 1:sibling_position + window_size + 1]
        )
        context = []
        for index in context_indices:
            unit = units[index]
            content = self._format_merged_component_content(
                unit.get("components", []),
                target_language,
            ).strip()
            placeholder = self._localize_merged_forward_placeholder(
                unit.get("placeholder"),
                target_language,
            )
            if placeholder:
                content = f"{placeholder}\n{content}" if content else placeholder
            if not content:
                continue
            context.append({
                "sender": "",
                "content": f"{self._merged_forward_header(unit, target_language)}: {content}",
            })
        return context

    def _format_merged_component_content(
        self,
        components,
        target_language: str = "Chinese",
    ) -> str:
        """Format merged-forward components and keep unsupported types visible."""
        renderable_components = []
        for component in components:
            kind = self._component_kind(component)
            if kind in {"face", "mface", "marketface", "market_face", "superface", "super_face"}:
                renderable_components.append(
                    Plain(text=self._merged_forward_label("emoji", target_language))
                )
                continue
            if not isinstance(component, dict):
                renderable_components.append(component)
                continue

            payload = component.get("data") if isinstance(component.get("data"), dict) else component
            if kind == "plain":
                renderable_components.append(Plain(text=str(payload.get("text") or "")))
            elif kind == "at":
                qq = payload.get("qq") or payload.get("user_id")
                renderable_components.append(
                    Plain(
                        text=f"<@{qq}>"
                        if qq
                        else self._merged_forward_label("mention", target_language)
                    )
                )
            elif kind == "image":
                source = payload.get("url") or payload.get("file") or payload.get("path")
                renderable_components.append(
                    Plain(
                        text=str(source)
                        if source
                        else self._merged_forward_label("image", target_language)
                    )
                )
            elif kind == "file":
                name = payload.get("name") or self._merged_forward_label("file", target_language)
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
        handled = {
            "plain",
            "at",
            "image",
            "file",
            "face",
            "mface",
            "marketface",
            "market_face",
            "superface",
            "super_face",
        }
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
                original_message = (
                    "原消息"
                    if self._is_chinese_language(target_language)
                    else "Original message"
                )
                extra_lines.append(
                    f"{self._merged_forward_label('quote', target_language)} "
                    f"{quote_text or original_message}"
                )
                continue
            if kind in {"record", "audio"}:
                source = (
                    getattr(component, "url", None)
                    or getattr(component, "file", None)
                    or getattr(component, "path", None)
                )
                extra_lines.append(
                    f"{self._merged_forward_label('audio', target_language)}({source})"
                    if isinstance(source, str) and source.startswith("http")
                    else self._merged_forward_label("audio", target_language)
                )
                continue
            if kind == "video":
                source = (
                    getattr(component, "url", None)
                    or getattr(component, "file", None)
                    or getattr(component, "path", None)
                )
                extra_lines.append(
                    f"{self._merged_forward_label('video', target_language)}({source})"
                    if isinstance(source, str) and source.startswith("http")
                    else self._merged_forward_label("video", target_language)
                )
                continue

            logger.warning(f"[MergedForward] 不支持的消息组件类型: {component.__class__.__name__}")
            extra_lines.append(
                f"[{self._merged_forward_label('unsupported', target_language)}: "
                f"{component.__class__.__name__}]"
            )

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

    def _merged_forward_target_language(self, rule: dict | None) -> str:
        if not self._is_forward_record_translation_enabled(rule):
            return "Chinese"
        translation = rule.get("translation", {}) if isinstance(rule, dict) else {}
        target_language = translation.get("target_language") if isinstance(translation, dict) else None
        return str(target_language or "Chinese").strip() or "Chinese"

    async def _translate_forward_record_components(
        self,
        event,
        components: list,
        rule: dict,
        translation_context: list[dict[str, str]] | None = None,
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
                translated_text = await self._translate_message(
                    event,
                    original_text,
                    rule,
                    background_context=translation_context,
                )
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
        translation_context: list[dict[str, str]] | None = None,
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
        components = await self._translate_forward_record_components(
            event,
            components,
            rule,
            translation_context,
        )
        target_language = self._merged_forward_target_language(rule)
        raw_content = self._format_merged_component_content(components, target_language)
        raw_content = self._restore_translation_literals(raw_content, protected_mentions)
        placeholder = self._localize_merged_forward_placeholder(
            unit.get("placeholder"),
            target_language,
        )
        if placeholder:
            raw_content = f"{placeholder}\n{raw_content}" if raw_content else placeholder

        header = self._merged_forward_header(unit, target_language)
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
