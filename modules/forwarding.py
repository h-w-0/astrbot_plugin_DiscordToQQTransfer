"""AstrBot event orchestration and platform-neutral forwarding flow."""

import asyncio

import aiohttp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..webhook import DiscordWebhookManager
from .storage import _classify_error


class ForwardingMixin:
    """Own the event hook, target ordering and per-rule orchestration."""

    async def _ensure_discord_webhook(self, target_umo: str) -> bool:
        """Create and cache a Discord webhook for a configured target."""
        if "discord" not in target_umo.lower():
            return False

        if await self.store.get_webhook_url(target_umo):
            return True

        parts = target_umo.split(":")
        channel_id = parts[2] if len(parts) >= 3 else parts[1] if len(parts) == 2 else ""
        if not channel_id:
            return False

        webhook_url = await self.webhook_manager.create_webhook_for_channel(int(channel_id))
        if not webhook_url:
            return False

        await self.store.set_webhook_url(target_umo, webhook_url)
        return True

    async def initialize(self):
        """Warm the Discord client and prepare configured Discord webhooks."""
        client = self.webhook_manager.get_discord_client()
        if client:
            logger.info("MsgTransfer: Discord 客户端已缓存")
        else:
            logger.info("MsgTransfer: Discord 客户端未就绪（非 Discord 环境或 py-cord 未安装）")

        configured_targets = {
            rule["target_umo"]
            for rule in self._get_config_forward_rules().values()
            if "discord" in rule["target_umo"].lower()
        }
        for target_umo in configured_targets:
            try:
                if not await self._ensure_discord_webhook(target_umo):
                    logger.warning(f"配置规则目标未能创建 Discord Webhook: {target_umo}")
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError) as exc:
                logger.warning(f"配置规则目标创建 Discord Webhook 失败 {target_umo}: {exc}")

        logger.info("MsgTransfer plugin init OK")

    async def _forward_message(self, event: AstrMessageEvent):
        """Forward one event to every matching target in configured order."""
        source_umo = ""
        output_slots: dict[str, tuple[asyncio.Future | None, asyncio.Future]] = {}
        completed_targets: set[str] = set()
        try:
            if self._is_notice_event(event):
                logger.debug("忽略 notice 事件，不参与消息转发")
                return

            source_umo = str(event.unified_msg_origin)
            rules = await self._list_forward_rules(source_umo)
            if not rules:
                return

            # Reserve each destination before any LLM or I/O work so concurrent
            # source channels still serialize output to the same target.
            for rule in rules.values():
                target_umo = str(rule["target_umo"])
                if target_umo not in output_slots:
                    output_slots[target_umo] = self._reserve_target_output_slot(target_umo)

            raw_message_chain = event.get_messages()
            try:
                # Resolve remote Forward(id) components once so every target
                # receives the same ordered snapshot of the QQ record.
                message_chain = await self._resolve_merged_forward_message(
                    event,
                    raw_message_chain,
                )
            except Exception as exc:
                logger.error(
                    f"[MergedForward] 远程聊天记录解析异常，继续使用原始消息链: {exc}",
                    exc_info=True,
                )
                message_chain = raw_message_chain
            platform = event.get_platform_name()
            if platform == "discord":
                discord_msg_id = event.message_obj.message_id
                if discord_msg_id:
                    message_text = DiscordWebhookManager.format_message_content(message_chain)
                    if message_text:
                        await self.store.add_forward_log(
                            str(discord_msg_id),
                            message_text,
                            event.get_sender_id(),
                        )

            for rule_id, rule in rules.items():
                target_umo = str(rule["target_umo"])
                output_predecessor, output_completion = output_slots[target_umo]
                try:
                    await self._forward_single_rule(
                        event,
                        rule,
                        rule_id,
                        source_umo,
                        message_chain,
                        output_predecessor=output_predecessor,
                    )
                finally:
                    self._complete_target_output_slot(
                        target_umo,
                        output_predecessor,
                        output_completion,
                    )
                    completed_targets.add(target_umo)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, KeyError) as exc:
            logger.error(f"❌ 转发逻辑异常: {exc}", exc_info=True)
        finally:
            for target_umo, (output_predecessor, output_completion) in output_slots.items():
                if target_umo not in completed_targets:
                    self._complete_target_output_slot(
                        target_umo,
                        output_predecessor,
                        output_completion,
                    )

    def _reserve_target_output_slot(
        self,
        target_umo: str,
    ) -> tuple[asyncio.Future | None, asyncio.Future]:
        output_tails = getattr(self, "_target_output_tails", None)
        if output_tails is None:
            output_tails = {}
            self._target_output_tails = output_tails

        output_completion = asyncio.get_running_loop().create_future()
        output_predecessor = output_tails.get(target_umo)
        output_tails[target_umo] = output_completion
        return output_predecessor, output_completion

    def _complete_target_output_slot(
        self,
        target_umo: str,
        output_predecessor: asyncio.Future | None,
        output_completion: asyncio.Future,
    ) -> None:
        def complete_after_predecessor(_completed_predecessor=None) -> None:
            if not output_completion.done():
                output_completion.set_result(None)

            output_tails = getattr(self, "_target_output_tails", None)
            if output_tails and output_tails.get(target_umo) is output_completion:
                del output_tails[target_umo]

        if output_predecessor is not None and not output_predecessor.done():
            output_predecessor.add_done_callback(complete_after_predecessor)
            return

        complete_after_predecessor()

    @staticmethod
    async def _wait_for_target_output(output_predecessor: asyncio.Future | None) -> None:
        if output_predecessor is not None:
            await output_predecessor

    @staticmethod
    def _is_notice_event(event) -> bool:
        """Return whether a OneBot raw payload is a notice event."""
        message_obj = getattr(event, "message_obj", None)
        raw_message = getattr(message_obj, "raw_message", None)
        if isinstance(raw_message, dict):
            post_type = raw_message.get("post_type")
        else:
            post_type = getattr(raw_message, "post_type", None)
        return isinstance(post_type, str) and post_type.lower() == "notice"

    async def _forward_single_rule(
        self,
        event: AstrMessageEvent,
        rule: dict,
        rule_id: str,
        source_umo: str,
        message_chain,
        output_predecessor: asyncio.Future | None = None,
    ):
        """Process one normalized forwarding rule."""
        try:
            platform = event.get_platform_name()
            if platform in ["aiocqhttp", "qqofficial"]:
                qq_id = event.get_sender_id()
                qq_name = event.get_sender_name()
                if await self.store.update_mapping(qq_id, qq_name):
                    logger.info(f"转发时已更新QQ号 {qq_id} 的名称: {qq_name}")

            target = rule["target_umo"]
            is_merged_forward = self._is_merged_forward_message(message_chain)
            message_text = (
                self._format_merged_forward_text(message_chain)
                if is_merged_forward
                else DiscordWebhookManager.format_message_content(message_chain)
            )
            content_safety = rule.get("content_safety", {})
            safety_value = (
                content_safety.get("enabled")
                if isinstance(content_safety, dict)
                else content_safety
            )
            if self._coerce_config_bool(safety_value, False):
                translation_enabled, safety_output_language = self._get_safety_output_language(rule)
                allowed, safety_reason = await self._passes_llm_safety_check(
                    event,
                    message_text,
                    target,
                    safety_output_language,
                    translation_enabled,
                )
                if not allowed:
                    logger.warning(f"转发 #{rule_id} 被内容安全筛查拦截: {target}")
                    await self._wait_for_target_output(output_predecessor)
                    await self._reply_safety_block(
                        event,
                        target,
                        safety_reason,
                        safety_output_language,
                    )
                    return

            webhook_url = await self.store.get_webhook_url(target)
            if webhook_url:
                if is_merged_forward:
                    await self._forward_merged_forward_with_webhook(
                        event,
                        target,
                        message_chain,
                        rule_id,
                        webhook_url,
                        rule,
                        output_predecessor,
                    )
                elif output_predecessor is None:
                    await self._forward_with_webhook(
                        event,
                        target,
                        message_chain,
                        rule_id,
                        webhook_url,
                        rule,
                    )
                else:
                    await self._forward_with_webhook(
                        event,
                        target,
                        message_chain,
                        rule_id,
                        webhook_url,
                        rule,
                        output_predecessor,
                    )
                return

            # Non-webhook targets use AstrBot's platform/session send path.
            try:
                sender_name = event.get_sender_name()
                source_platform_name = event.get_platform_name()
                message_text = DiscordWebhookManager.format_message_content(message_chain)
                if message_text:
                    full_text = f"[转发] {sender_name} ({source_platform_name})​: {message_text}"
                else:
                    full_text = f"[转发] {sender_name} ({source_platform_name})​"

                translated = await self._translate_message(
                    event,
                    message_text or full_text,
                    rule,
                )
                if translated:
                    message_text = translated
                    full_text = f"[转发] {sender_name} ({source_platform_name})​: {translated}"

                chain = await self._build_discord_reply_chain(
                    event,
                    source_platform_name,
                    sender_name,
                    message_text,
                    full_text,
                )
                await self._wait_for_target_output(output_predecessor)
                sent, sent_result = await self._send_message_with_result(target, chain)
                if sent:
                    await self._record_discord_to_target_mapping(
                        event,
                        sent_result,
                        source_platform_name,
                    )
                    logger.info(f"已转发 #{rule_id} -> {target}")
                else:
                    logger.warning(f"转发 #{rule_id} 未找到目标平台适配器: {target}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.error(f"通过 AstrBot 转发 #{rule_id} 网络错误: {exc}")
            except (OSError, ValueError, KeyError) as exc:
                logger.error(f"通过 AstrBot 转发 #{rule_id} 失败: {exc}")
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            logger.error(f"❌ 处理规则 #{rule_id} 时发生异常: {exc}")

    async def terminate(self):
        """Close plugin-owned network resources during unload."""
        await self.webhook_manager.close()
        logger.info("MsgTransfer plugin terminated")


__all__ = ["ForwardingMixin"]
