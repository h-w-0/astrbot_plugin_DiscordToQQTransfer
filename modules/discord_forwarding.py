"""Discord webhook and reply-specific forwarding operations."""

import asyncio
import time

import aiohttp
from astrbot.api import logger

try:
    from astrbot.api.message_components import Plain, Reply, At, MessageChain
except ImportError:
    from astrbot.core.message.components import Plain, Reply, At
    from astrbot.core.message.message_event_result import MessageChain

from ..webhook import DiscordWebhookManager
from .storage import _classify_error


class DiscordForwardingMixin:
    """Discord-specific rendering, reply resolution and webhook sending."""

    async def _forward_merged_forward_with_webhook(
        self,
        event,
        target_umo: str,
        message_chain,
        rule_id: str,
        webhook_url: str,
        rule: dict | None = None,
        output_predecessor: asyncio.Future | None = None,
    ) -> bool:
        """Create a thread and send expanded merged-forward nodes in order."""
        channel_id = self._discord_channel_id(target_umo)
        if channel_id is None:
            logger.error(f"[MergedForward] 合并转发 #{rule_id} 的 Discord 目标频道无效: {target_umo}")
            return False

        sender_name = self._event_value(event, "get_sender_name", "QQ用户")
        sender_id = self._event_value(event, "get_sender_id")
        source_platform = self._event_value(event, "get_platform_name", "aiocqhttp")
        message_id = getattr(getattr(event, "message_obj", None), "message_id", None)
        rule_config = rule or {}
        target_language = self._merged_forward_target_language(rule_config)
        thread_label = self._merged_forward_label("merged_forward", target_language)
        thread_name = f"{thread_label} - {sender_name} - {message_id or time.strftime('%m%d-%H%M%S')}"

        try:
            thread = await self.webhook_manager.create_thread_for_channel(channel_id, thread_name)
        except Exception as exc:
            logger.error(f"[MergedForward] 创建 Thread 异常 #{rule_id}: {_classify_error(exc)}")
            return False

        thread_id = self._thread_id(thread)
        if not thread_id:
            logger.error(f"[MergedForward] 创建 Thread 失败 #{rule_id}: 未返回 Thread ID")
            return False
        logger.info(f"已创建 QQ 合并转发 Thread #{rule_id} -> {target_umo} (thread={thread_id})")

        try:
            mapping_getter = getattr(self.store, "load_mappings", None)
            mapping = await mapping_getter() if callable(mapping_getter) else {}
            if not isinstance(mapping, dict):
                mapping = {}
        except Exception as exc:
            logger.warning(f"读取QQ名称映射失败，合并转发中的艾特将使用QQ号: {exc}")
            mapping = {}

        username = DiscordWebhookManager.build_virtual_username(sender_name, source_platform)
        avatar_url = DiscordWebhookManager.get_avatar_url(source_platform, sender_id)
        first_discord_msg_id = None
        all_sent = True
        units = self._build_merged_forward_units(message_chain)
        use_forward_translation_context = self._is_forward_record_translation_enabled(rule_config)

        for unit_index, unit in enumerate(units, start=1):
            try:
                translation_context = (
                    self._build_merged_forward_translation_context(
                        units,
                        unit_index - 1,
                        target_language,
                    )
                    if use_forward_translation_context
                    else None
                )
                content, embeds, files = await self._prepare_merged_forward_unit(
                    event,
                    unit,
                    rule_config,
                    mapping,
                    translation_context,
                )
                chunks = self._split_discord_content(content)
                for chunk_index, chunk in enumerate(chunks):
                    await self._wait_for_target_output(output_predecessor)
                    sent_id = await self.webhook_manager.send_webhook_message(
                        webhook_url=webhook_url,
                        username=username,
                        avatar_url=avatar_url,
                        content=chunk,
                        embeds=embeds if chunk_index == 0 else None,
                        files=files if chunk_index == 0 else None,
                        thread_id=thread_id,
                    )
                    if not sent_id:
                        all_sent = False
                        logger.error(
                            f"[MergedForward] Thread 消息发送失败 #{rule_id} "
                            f"(节点={unit_index}, 分片={chunk_index + 1})"
                        )
                    elif first_discord_msg_id is None:
                        first_discord_msg_id = str(sent_id)
            except Exception as exc:
                all_sent = False
                logger.error(
                    f"[MergedForward] 处理 Thread 节点失败 #{rule_id} "
                    f"(节点={unit_index}): {_classify_error(exc)}",
                    exc_info=True,
                )

        if first_discord_msg_id and message_id:
            try:
                await self.store.set_msg_mapping(
                    str(message_id),
                    first_discord_msg_id,
                    sender_id,
                    sender_name,
                    forwarded_content=self._format_merged_forward_text(message_chain)[:2000],
                )
            except Exception as exc:
                logger.error(f"保存合并转发消息映射失败(不影响发送): {exc}")

        if all_sent:
            logger.info(f"已完成 QQ 合并转发 #{rule_id} -> Discord Thread {thread_id}")
        else:
            logger.warning(f"QQ 合并转发 #{rule_id} 部分消息发送失败，已继续处理其余节点")
        return all_sent

    async def _build_discord_reply_chain(
        self,
        event,
        source_platform_name,
        sender_name,
        msg_text,
        full_text,
    ):
        """Build the platform message chain used for non-webhook forwarding."""
        chain_parts = []
        if source_platform_name == "discord":
            raw_message = getattr(event.message_obj, "raw_message", None)
            if raw_message:
                reference = getattr(raw_message, "reference", None)
                if reference and reference.message_id:
                    original_qq_id = await self.store.find_qq_msg_id_by_discord_id(
                        str(reference.message_id)
                    )
                    if original_qq_id:
                        meta = await self.store.get_msg_meta(original_qq_id)
                        if meta and meta.get("origin", "qq") == "qq":
                            chain_parts.append(Reply(id=original_qq_id))
                            chain_parts.append(
                                Plain(text=f"[转发] {sender_name} ({source_platform_name}):")
                            )
                            chain_parts.append(At(qq=meta["user_id"]))
                            if msg_text:
                                chain_parts.append(Plain(text=f" {msg_text}"))
                        else:
                            chain_parts.append(Reply(id=original_qq_id))

        if not chain_parts:
            chain_parts.append(Plain(text=full_text))
        elif not any(isinstance(component, At) for component in chain_parts):
            chain_parts.append(Plain(text=full_text))
        chain = MessageChain()
        chain.chain = chain_parts
        return chain

    @staticmethod
    def _extract_message_id_from_send_result(result) -> str | None:
        """Best-effort extraction of a sent message ID from platform results."""
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
                nested_id = DiscordForwardingMixin._extract_message_id_from_send_result(
                    result.get(key)
                )
                if nested_id:
                    return nested_id
            return None

        for attribute in ("message_id", "msg_id", "id"):
            value = getattr(result, attribute, None)
            if value:
                return str(value)

        if isinstance(result, (list, tuple)):
            for item in result:
                nested_id = DiscordForwardingMixin._extract_message_id_from_send_result(item)
                if nested_id:
                    return nested_id

        return None

    async def _send_message_with_result(
        self,
        target: str,
        chain: MessageChain,
    ) -> tuple[bool, object | None]:
        """Send through a matching platform instance while preserving the result."""
        if self._message_session_type is None:
            sent = await self.context.send_message(target, chain)
            return bool(sent), sent

        session = self._message_session_type.from_str(target) if isinstance(target, str) else target
        platform_manager = getattr(self.context, "platform_manager", None)
        platform_insts = (
            getattr(platform_manager, "platform_insts", [])
            if platform_manager
            else []
        )
        for platform in platform_insts:
            meta = platform.meta()
            if meta.name != session.platform_name:
                continue
            result = await platform.send_by_session(session, chain)
            return True, result

        sent = await self.context.send_message(target, chain)
        return bool(sent), sent

    async def _record_discord_to_target_mapping(
        self,
        event,
        sent_result,
        source_platform_name: str,
    ):
        """Record the target platform ID for a Discord source message."""
        if source_platform_name != "discord":
            return

        discord_msg_id = getattr(event.message_obj, "message_id", None)
        if not discord_msg_id:
            return

        sent_msg_id = self._extract_message_id_from_send_result(sent_result)
        if not sent_msg_id:
            logger.debug(
                "Discord→QQ 转发成功，但无法从发送结果提取目标消息 ID，保留文本回退匹配"
            )
            return

        try:
            await self.store.set_msg_mapping(
                str(sent_msg_id),
                str(discord_msg_id),
                event.get_sender_id(),
                event.get_sender_name(),
                origin="discord",
            )
        except Exception as exc:
            logger.error(f"保存 Discord→QQ 消息映射失败(不影响发送): {exc}")

    async def _resolve_reply_target(
        self,
        reply_to_qq_id,
        quote_text,
        target_umo,
        prefer_forwarded_content: bool = True,
    ):
        """Resolve a QQ reply to a Discord message, sender, jump URL and quote text."""
        reply_to_discord_id = None
        discord_sender_id = None
        forwarded_quote_text = None
        if reply_to_qq_id:
            reply_to_discord_id = await self.store.get_msg_mapping(reply_to_qq_id)
            if reply_to_discord_id:
                meta = await self.store.get_msg_meta(reply_to_qq_id)
                if meta:
                    if prefer_forwarded_content:
                        forwarded_quote_text = meta.get("forwarded_content") or None
                    if meta.get("origin") == "discord":
                        discord_sender_id = meta.get("user_id")

        if reply_to_discord_id is None and quote_text:
            forwarded_discord_id = await self.store.find_forward_log_by_content(quote_text)
            if forwarded_discord_id:
                reply_to_discord_id = forwarded_discord_id
                discord_sender_id = await self.store.get_forward_entry_sender(
                    forwarded_discord_id
                )

        jump_url = None
        if reply_to_discord_id:
            channel_id = self._discord_channel_id(target_umo)
            if channel_id:
                try:
                    client = self.webhook_manager.get_discord_client()
                    if client:
                        channel = await client.fetch_channel(channel_id)
                        if hasattr(channel, "guild") and channel.guild:
                            guild_id = channel.guild.id
                            jump_url = (
                                f"https://discord.com/channels/{guild_id}/"
                                f"{channel_id}/{reply_to_discord_id}"
                            )
                        if (
                            prefer_forwarded_content
                            and not forwarded_quote_text
                            and hasattr(channel, "fetch_message")
                        ):
                            try:
                                referenced_message = await channel.fetch_message(
                                    int(reply_to_discord_id)
                                )
                                forwarded_quote_text = (
                                    getattr(referenced_message, "content", None) or None
                                )
                            except Exception as exc:
                                logger.debug(
                                    f"读取 Discord 被引用消息失败，继续使用原引用文本: {exc}"
                                )
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError) as exc:
                    logger.warning(f"构建 Discord jump URL 失败: {exc}")

        return (
            reply_to_discord_id,
            discord_sender_id,
            jump_url,
            forwarded_quote_text,
        )

    async def _translate_webhook_quote(
        self,
        event,
        quote_text: str,
        rule: dict | None,
    ) -> str:
        """Translate a QQ quote using the current forwarding rule."""
        original_text = str(quote_text or "")
        if not original_text.strip() or not self._is_translation_enabled_for_rule(rule):
            return original_text

        try:
            translated_text = await self._translate_message(
                event,
                original_text,
                rule,
                translation_session_id=(
                    f"{self._build_translation_session_id(event)}:quote"
                ),
            )
        except Exception as exc:
            logger.warning(f"引用文本翻译失败，回退原文: {exc}")
            return original_text
        return translated_text or original_text

    async def _forward_with_webhook(
        self,
        event,
        target_umo: str,
        message_chain,
        rule_id: str,
        webhook_url: str,
        rule: dict = None,
        output_predecessor: asyncio.Future | None = None,
    ) -> bool:
        try:
            sender_name = event.get_sender_name()
            sender_id = event.get_sender_id()
            source_platform = event.get_platform_name()
            self_id = event.get_self_id()
            mapping = await self.store.load_mappings()

            quote_text, quote_sender, reply_to_qq_id = self._extract_quote_info(message_chain)
            quote_text, quote_sender, discord_sender_name = self._resolve_forward_quote(
                quote_text,
                quote_sender,
            )
            has_original_quote_text = bool(str(quote_text or "").strip())

            translation_enabled = self._is_translation_enabled_for_rule(rule)
            (
                reply_to_discord_id,
                discord_sender_id,
                jump_url,
                forwarded_quote_text,
            ) = await self._resolve_reply_target(
                reply_to_qq_id,
                quote_text,
                target_umo,
                prefer_forwarded_content=translation_enabled,
            )
            if forwarded_quote_text and not has_original_quote_text:
                quote_text = forwarded_quote_text

            protected_mentions: dict[str, str] = {}
            new_chain = self._replace_ats(
                message_chain,
                discord_sender_id,
                discord_sender_name,
                mapping,
                self_id,
                protected_mentions,
            )

            virtual_username = DiscordWebhookManager.build_virtual_username(
                sender_name,
                source_platform,
            )
            avatar_url = DiscordWebhookManager.get_avatar_url(source_platform, sender_id)
            image_urls = DiscordWebhookManager.extract_images(new_chain)
            local_images = DiscordWebhookManager.extract_local_image_paths(new_chain)
            raw_content = DiscordWebhookManager.format_message_content(
                new_chain,
                skip_images=True,
            )

            translated = None
            if rule is not None:
                translated = await self._translate_message(event, raw_content, rule)
            if translated:
                raw_content = translated
            raw_content = self._restore_translation_literals(raw_content, protected_mentions)
            if has_original_quote_text:
                # Stored content can belong to another target language. Translate
                # the original QQ quote for this rule instead of reusing that cache.
                quote_text = await self._translate_webhook_quote(event, quote_text, rule)

            content = self._build_webhook_quote(
                raw_content,
                reply_to_discord_id,
                jump_url,
                quote_text,
                quote_sender,
            )
            embeds = [{"image": {"url": url}} for url in image_urls[:10]]
            if not content and not embeds and not local_images:
                content = "[图片]"

            await self._wait_for_target_output(output_predecessor)
            discord_msg_id = await self.webhook_manager.send_webhook_message(
                webhook_url=webhook_url,
                username=virtual_username,
                avatar_url=avatar_url,
                content=content,
                embeds=embeds if embeds else None,
                files=local_images[:10] if local_images else None,
            )

            if discord_msg_id:
                qq_msg_id = event.message_obj.message_id
                if qq_msg_id:
                    qq_user_id = event.get_sender_id()
                    qq_user_name = event.get_sender_name()
                    try:
                        await self.store.set_msg_mapping(
                            qq_msg_id,
                            discord_msg_id,
                            qq_user_id,
                            qq_user_name,
                            forwarded_content=raw_content,
                        )
                    except Exception as exc:
                        logger.error(f"保存消息映射 #{rule_id} 失败(不影响发送): {exc}")
                return True

            return False
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error(f"❌ Webhook网络错误 #{rule_id}: {exc}")
            return False
        except Exception as exc:
            logger.error(f"❌ Webhook转发异常 #{rule_id}: {_classify_error(exc)}")
            return False


__all__ = ["DiscordForwardingMixin"]
