"""AstrBot entrypoint for the Discord/QQ message transfer plugin."""

from collections import OrderedDict

import astrbot.api.star as star
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context

from .modules.config import ConfigMixin
from .modules.discord_forwarding import DiscordForwardingMixin
from .modules.forwarding import ForwardingMixin
from .modules.llm import (
    LlmMixin,
    _AsyncOpenAI,
    _OpenAIError,
    llm_provider_error_types,
)
from .modules.message_processing import MessageProcessingMixin
from .modules.storage import (
    MsgTransferStore,
    _classify_error,
    async_read_json,
    async_write_json,
)
from .modules.translation import (
    LANG_CODE_MAP,
    TranslationMixin,
    _detect_lang,
    detect_source_language,
)
from .sensitive_lexicon import load_bundled_sensitive_lexicon
from .webhook import DiscordWebhookManager

try:
    from astrbot.core.platform.astr_message_event import MessageSesion
except ImportError:
    MessageSesion = None


class MsgTransfer(
    ForwardingMixin,
    DiscordForwardingMixin,
    MessageProcessingMixin,
    LlmMixin,
    TranslationMixin,
    ConfigMixin,
    star.Star,
):
    """AstrBot plugin entrypoint composed from domain-specific mixins."""

    _message_session_type = MessageSesion

    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.plugin_config = config

        self.data_dir = star.StarTools.get_data_dir(
            "astrbot_plugin_DiscordToQQTransfer"
        )
        self.forward_log_file = self.data_dir / "forward_log.json"
        self.webhook_file = self.data_dir / "webhooks.json"
        self.mapping_file = self.data_dir / "mappings.json"
        self.msg_mapping_file = self.data_dir / "msg_mapping.json"

        self.store = MsgTransferStore(
            self.webhook_file,
            self.mapping_file,
            self.msg_mapping_file,
            self.forward_log_file,
        )
        self.webhook_manager = DiscordWebhookManager(context)
        self._target_output_tails = {}
        self._translation_contexts: OrderedDict = OrderedDict()

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def forward_message(self, event: AstrMessageEvent):
        """AstrBot event registration shim; implementation lives in forwarding.py."""
        return await self._forward_message(event)

    async def initialize(self):
        """Delegate plugin startup to the forwarding domain module."""
        return await ForwardingMixin.initialize(self)

    async def terminate(self):
        """Delegate resource cleanup to the forwarding domain module."""
        return await ForwardingMixin.terminate(self)

    @staticmethod
    def _detect_source_language(text: str) -> str:
        """Keep the historical main-module detector override available to callers."""
        return detect_source_language(text, detector=_detect_lang)


__all__ = [
    "MsgTransfer",
    "MsgTransferStore",
    "DiscordWebhookManager",
    "async_read_json",
    "async_write_json",
    "_classify_error",
    "_AsyncOpenAI",
    "_OpenAIError",
    "_detect_lang",
    "LANG_CODE_MAP",
    "llm_provider_error_types",
    "load_bundled_sensitive_lexicon",
    "logger",
]
