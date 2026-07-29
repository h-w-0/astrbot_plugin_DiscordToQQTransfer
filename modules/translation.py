"""Translation prompts, language detection and short-term context."""

import asyncio
import re
import time
from collections import OrderedDict, deque

import aiohttp

from .config import DEFAULT_RECENT_CONTEXT_COUNT, MAX_RECENT_CONTEXT_COUNT

try:
    from langdetect import detect as _detect_lang
except ImportError:
    _detect_lang = None


LANG_CODE_MAP = {
    "af": "Afrikaans", "bg": "Bulgarian", "ca": "Catalan", "cy": "Welsh",
    "da": "Danish", "et": "Estonian", "fi": "Finnish", "hr": "Croatian",
    "hu": "Hungarian", "lt": "Lithuanian", "lv": "Latvian",
    "mk": "Macedonian", "no": "Norwegian", "ro": "Romanian",
    "sk": "Slovak", "sl": "Slovenian", "so": "Somali", "sq": "Albanian",
    "sv": "Swedish", "sw": "Swahili",
    "zh": "Chinese", "zh-cn": "Chinese", "zh-tw": "Traditional Chinese",
    "en": "English", "fr": "French", "pt": "Portuguese", "es": "Spanish",
    "ja": "Japanese", "tr": "Turkish", "ru": "Russian", "ar": "Arabic",
    "ko": "Korean", "th": "Thai", "it": "Italian", "de": "German",
    "vi": "Vietnamese", "ms": "Malay", "id": "Indonesian", "tl": "Filipino",
    "hi": "Hindi", "pl": "Polish", "cs": "Czech", "nl": "Dutch",
    "km": "Khmer", "my": "Burmese", "fa": "Persian", "gu": "Gujarati",
    "ur": "Urdu", "te": "Telugu", "mr": "Marathi", "he": "Hebrew",
    "bn": "Bengali", "ta": "Tamil", "uk": "Ukrainian", "bo": "Tibetan",
    "kk": "Kazakh", "mn": "Mongolian", "ug": "Uyghur", "yue": "Cantonese",
}

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff\uff66-\uff9f]")
_HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7ff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u052f]")
_UKRAINIAN_HINT_RE = re.compile(r"[\u0404\u0406\u0407\u0490\u0454\u0456\u0457\u0491]")
_BELARUSIAN_HINT_RE = re.compile(r"[\u040e\u045e]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_DISCORD_MARKUP_RE = re.compile(r"<(?:(?:@!?|@&|#)\d+|a?:[^:>]+:\d+)>")
_TRANSLATION_LITERAL_RE = re.compile(
    r"__ASTRBOT_AT_\d+__|<(?:(?:@!?|@&|#)\d+|a?:[^:>]+:\d+)>",
    re.IGNORECASE,
)
_SHORT_ASCII_ENGLISH_MAX_LETTERS = 12
_SHORT_CYRILLIC_RUSSIAN_MAX_LETTERS = 12

try:
    from openai import OpenAIError as _OpenAIError
except ImportError:
    _OpenAIError = None

if _OpenAIError is None:
    llm_provider_error_types = (
        asyncio.TimeoutError,
        aiohttp.ClientError,
        OSError,
        ValueError,
        RuntimeError,
    )
else:
    llm_provider_error_types = (
        asyncio.TimeoutError,
        aiohttp.ClientError,
        OSError,
        ValueError,
        RuntimeError,
        _OpenAIError,
    )


def detect_source_language(text: str, detector=None) -> str:
    """Detect a source language while handling short and markup-only messages."""
    if not text or not text.strip():
        return "Unknown"

    sample = _URL_RE.sub(" ", text)
    sample = _TRANSLATION_LITERAL_RE.sub(" ", sample)
    sample = " ".join(sample.split())
    if not sample or not any(char.isalpha() for char in sample):
        return "Unknown"

    if _KANA_RE.search(sample):
        return "Japanese"
    if _HANGUL_RE.search(sample):
        return "Korean"
    if _HAN_RE.search(sample):
        return "Chinese"

    cyrillic_letters = _CYRILLIC_RE.findall(sample)
    has_only_cyrillic_letters = bool(cyrillic_letters) and len(cyrillic_letters) == sum(
        char.isalpha() for char in sample
    )
    if has_only_cyrillic_letters:
        if _BELARUSIAN_HINT_RE.search(sample):
            return "Belarusian"
        if _UKRAINIAN_HINT_RE.search(sample):
            return "Ukrainian"
        if len(cyrillic_letters) <= _SHORT_CYRILLIC_RUSSIAN_MAX_LETTERS:
            return "Russian"

    ascii_letters = _ASCII_LETTER_RE.findall(sample)
    has_only_ascii_letters = bool(ascii_letters) and all(
        ord(char) < 128 for char in sample if char.isalpha()
    )
    if has_only_ascii_letters and len(ascii_letters) <= _SHORT_ASCII_ENGLISH_MAX_LETTERS:
        return "English"

    if detector is None:
        detector = _detect_lang
    if detector is not None:
        try:
            code = detector(sample).lower()
            detected_language = LANG_CODE_MAP.get(code)
            if detected_language:
                return detected_language
        except Exception:
            pass

    if has_only_ascii_letters:
        return "English"
    if has_only_cyrillic_letters:
        return "Russian"
    return "Unknown"


class TranslationMixin:
    """Translation behavior shared by webhook and platform forwarding paths."""

    TRANSLATION_CONTEXT_SIZE = DEFAULT_RECENT_CONTEXT_COUNT
    MAX_TRANSLATION_CONTEXT_SIZE = MAX_RECENT_CONTEXT_COUNT
    MAX_TRANSLATION_CONTEXTS = 200

    @staticmethod
    def _is_chinese_language(language: str) -> bool:
        language_key = str(language or "").strip().lower().replace("_", "-")
        return language_key in {
            "chinese", "中文", "汉语", "漢語", "简体中文", "簡體中文",
            "繁体中文", "繁體中文", "zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant",
        }

    @staticmethod
    def _build_translation_session_id(event) -> str:
        """Use an isolated provider session for every translated message."""
        message_id = getattr(getattr(event, "message_obj", None), "message_id", None)
        if message_id:
            return f"msg_transfer_translate:{message_id}"
        return f"msg_transfer_translate:{event.unified_msg_origin}:{time.time_ns()}"

    async def _translate_message(
        self,
        event,
        msg_text: str,
        rule: dict,
        background_context: list[dict[str, str]] | None = None,
    ) -> str | None:
        """Translate a rule-enabled message, returning None on disabled or failed calls."""
        translation_config = self._get_llm_translation_config()
        if not translation_config.get("enabled"):
            return None

        rule_translation = rule.get("translation", {})
        if not isinstance(rule_translation, dict):
            return None
        if not self._coerce_config_bool(rule_translation.get("enabled"), False):
            return None

        target_language = str(rule_translation.get("target_language", "Chinese")).strip()
        if not msg_text or not msg_text.strip():
            return None

        recent_context = self._remember_translation_context(event, msg_text, rule)
        source_language = str(rule_translation.get("source_language", "")).strip()
        if not source_language:
            source_language = self._detect_source_language(msg_text)

        protected_text, protected_literals = self._protect_translation_literals(msg_text)
        try:
            context_messages = (
                list(background_context)
                if background_context is not None
                else (
                    recent_context
                    if translation_config.get("use_recent_context")
                    else []
                )
            )
            if context_messages:
                background_text = "\n".join(
                    f"{item['sender']}: {item['content']}" if item["sender"] else item["content"]
                    for item in context_messages
                )
                prompt = (
                    "[Background Information]\n"
                    f"{background_text}\n\n"
                    f"Please translate the following text into {target_language}, taking the provided "
                    "background information into consideration.\n\n"
                    "[Source Text]\n"
                    f"{protected_text}"
                )
            else:
                prompt = self._build_translation_prompt(target_language, protected_text)
            if protected_literals:
                prompt = (
                    "Placeholder tokens in the source text are literal text; copy them unchanged and "
                    "output only the translation.\n\n"
                    f"{prompt}"
                )
        except Exception as exc:
            from astrbot.api import logger

            logger.warning(f"翻译提示词模板替换失败: {exc}")
            return None

        # Hy-MT2-style providers receive all instructions in the user prompt.
        translation_config["system_prompt"] = ""
        provider_config = dict(translation_config)
        provider_config.pop("recent_context_count", None)
        try:
            response_text = await self._call_llm(
                prompt=prompt,
                cfg=provider_config,
                session_id=self._build_translation_session_id(event),
                umo=getattr(event, "unified_msg_origin", None),
                tag="翻译",
            )
            if response_text and response_text.strip():
                if self._is_translation_prompt_echo(response_text):
                    from astrbot.api import logger

                    logger.warning("LLM 翻译返回内容疑似包含内部提示词，回退原文")
                    return None
                prefix = self._format_translation_prefix(source_language, target_language)
                translated_text = self._restore_translation_literals(
                    response_text.strip(),
                    protected_literals,
                )
                return f"{prefix}{translated_text}"
            return None
        except llm_provider_error_types as exc:
            from astrbot.api import logger

            logger.warning(f"LLM 翻译失败: {exc}")
            return None

    @staticmethod
    def _is_translation_prompt_echo(response_text: str) -> bool:
        """Reject provider responses that expose translation instructions or context blocks."""
        normalized = str(response_text or "").strip().lower()
        if not normalized:
            return False
        return any(
            marker in normalized
            for marker in (
                "placeholder tokens exactly unchanged",
                "[background information]",
                "[source text]",
                "【背景信息】",
                "【待翻译文本】",
                "不要复述背景信息或提示词",
            )
        )

    @classmethod
    def _build_translation_prompt(cls, target_language: str, source_text: str) -> str:
        """Build the fixed translation instruction used by supported providers."""
        return (
            f"Translate the following text into {target_language}. Note that you should only output "
            f"the translated result without any additional explanation:\n\n{source_text}"
        )

    def _remember_translation_context(
        self,
        event,
        msg_text: str,
        rule: dict,
    ) -> list[dict[str, str]]:
        """Return configured prior messages and remember the current one once."""
        context_size = self._get_llm_translation_config().get(
            "recent_context_count",
            self.TRANSLATION_CONTEXT_SIZE,
        )
        try:
            context_size = min(self.MAX_TRANSLATION_CONTEXT_SIZE, max(0, int(context_size)))
        except (TypeError, ValueError):
            context_size = self.TRANSLATION_CONTEXT_SIZE

        contexts = getattr(self, "_translation_contexts", None)
        if contexts is None:
            contexts = OrderedDict()
            self._translation_contexts = contexts

        source_key = str(rule.get("source_umo") or getattr(event, "unified_msg_origin", ""))
        history = contexts.get(source_key)
        if history is None:
            history = deque(maxlen=self.MAX_TRANSLATION_CONTEXT_SIZE + 1)
            contexts[source_key] = history
            if len(contexts) > self.MAX_TRANSLATION_CONTEXTS:
                contexts.popitem(last=False)
        else:
            contexts.move_to_end(source_key)

        message_id = getattr(getattr(event, "message_obj", None), "message_id", None)
        message_key = str(message_id) if message_id is not None else f"event:{id(event)}"
        prior_messages = [
            {"sender": sender, "content": content}
            for stored_key, sender, content in history
            if stored_key != message_key
        ]
        recent_context = prior_messages[-context_size:] if context_size else []

        if not any(stored_key == message_key for stored_key, _sender, _content in history):
            sender_getter = getattr(event, "get_sender_name", None)
            sender = str(sender_getter() if callable(sender_getter) else "")
            history.append((message_key, sender, str(msg_text)[:1000]))

        return recent_context

    @staticmethod
    def _protect_translation_literals(text: str) -> tuple[str, dict[str, str]]:
        """Replace mentions with stable tokens so the model cannot translate them."""
        protected_literals: dict[str, str] = {}

        def replace_literal(match: re.Match) -> str:
            index = len(protected_literals)
            token = f"__ASTRBOT_KEEP_{index:04d}__"
            while token in text or token in protected_literals:
                index += 1
                token = f"__ASTRBOT_KEEP_{index:04d}__"
            protected_literals[token] = match.group(0)
            return token

        return _TRANSLATION_LITERAL_RE.sub(replace_literal, text), protected_literals

    @staticmethod
    def _restore_translation_literals(text: str, protected_literals: dict[str, str]) -> str:
        """Restore protected literals and retain any token the model dropped."""
        restored = text
        missing_literals = []
        for token, literal in protected_literals.items():
            restored, count = re.subn(
                re.escape(token),
                lambda _match, value=literal: value,
                restored,
                flags=re.IGNORECASE,
            )
            if count == 0:
                missing_literals.append(literal)
        return "".join(missing_literals) + restored

    @staticmethod
    def _detect_source_language(text: str) -> str:
        return detect_source_language(text)

    @staticmethod
    def _format_translation_prefix(source_language: str, target_language: str) -> str:
        """Build a Chinese or English translation prefix from the target language."""
        target_key = str(target_language or "").strip().lower().replace("_", "-")
        chinese_target = target_key in {
            "chinese", "中文", "zh", "zh-cn", "zh-tw", "简体中文", "繁体中文", "繁體中文",
        }

        source_key = str(source_language or "").strip().lower().replace("_", "-")
        if source_key in {"", "unknown", "auto", "und"}:
            return "(从原文翻译)" if chinese_target else "(Translated from original text)"
        if source_key in {
            "chinese", "中文", "zh", "zh-cn", "zh-tw", "简体中文", "繁体中文",
            "繁體中文", "traditional chinese",
        }:
            source_name = "中文" if chinese_target else "Chinese"
        elif source_key in {"english", "英语", "英文", "en", "en-us", "en-gb"}:
            source_name = "英文" if chinese_target else "English"
        elif source_key in {"russian", "俄语", "俄文", "ru", "русский"}:
            source_name = "俄文" if chinese_target else "Russian"
        else:
            source_name = str(source_language or "").strip()

        if chinese_target:
            return f"(从{source_name}翻译)"
        return f"(Translated from {source_name})"


__all__ = [
    "TranslationMixin",
    "LANG_CODE_MAP",
    "detect_source_language",
    "_detect_lang",
    "_TRANSLATION_LITERAL_RE",
    "llm_provider_error_types",
]
