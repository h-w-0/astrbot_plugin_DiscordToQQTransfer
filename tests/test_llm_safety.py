import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "astrbot_plugin_msg_transfer"
MAIN_MODULE_NAME = f"{PACKAGE_NAME}.main"

package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(REPO_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)
spec = importlib.util.spec_from_file_location(MAIN_MODULE_NAME, REPO_ROOT / "main.py")
if spec is None or spec.loader is None:
    raise ImportError("无法加载插件 main.py")
module = importlib.util.module_from_spec(spec)
sys.modules[MAIN_MODULE_NAME] = module
spec.loader.exec_module(module)
MsgTransfer = module.MsgTransfer


class DummyProvider:
    async def text_chat(self, **kwargs):
        raise openai.OpenAIError("thinking invalid")


class SuccessfulProvider:
    def __init__(self, completion_text='{"safe": true, "reason": "内容正常"}'):
        self.completion_text = completion_text
        self.calls = []

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(role="assistant", completion_text=self.completion_text)


class DummyContext:
    def __init__(self, provider=None):
        self.provider = provider or DummyProvider()

    def get_using_provider(self, umo=None):
        return self.provider


class LlmSafetyCheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_notice_event_is_ignored_before_forwarding(self):
        plugin = object.__new__(MsgTransfer)
        plugin._list_forward_rules = AsyncMock(
            side_effect=AssertionError("notice 事件不应查询转发规则")
        )
        event = SimpleNamespace(
            message_obj=SimpleNamespace(
                raw_message={"post_type": "notice", "notice_type": "group_recall"}
            )
        )

        await plugin.forward_message(event)

    async def test_message_event_still_enters_forwarding(self):
        plugin = object.__new__(MsgTransfer)
        plugin._list_forward_rules = AsyncMock(return_value={})
        event = SimpleNamespace(
            unified_msg_origin="aiocqhttp:GroupMessage:123456",
            message_obj=SimpleNamespace(raw_message={"post_type": "message"})
        )

        await plugin.forward_message(event)

        plugin._list_forward_rules.assert_awaited_once()

    async def test_openai_provider_error_follows_allow_on_error_config(self):
        plugin = object.__new__(MsgTransfer)
        plugin.context = DummyContext()
        plugin.plugin_config = {
            "llm_safety_check": {
                "enabled": True,
                "block_on_error": False,
                "timeout_seconds": 1,
            }
        }
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_id="discord-1"),
            unified_msg_origin="discord:channel:1",
            get_sender_name=lambda: "tester",
            get_sender_id=lambda: "user-1",
        )

        allowed, reason = await plugin._passes_llm_safety_check(event, "普通消息")

        self.assertTrue(allowed)
        self.assertEqual(reason, "安全审核失败或超时")

    async def test_openai_provider_error_follows_block_on_error_config(self):
        plugin = object.__new__(MsgTransfer)
        plugin.context = DummyContext()
        plugin.plugin_config = {
            "llm_safety_check": {
                "enabled": True,
                "block_on_error": True,
                "timeout_seconds": 1,
            }
        }
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_id="discord-2"),
            unified_msg_origin="discord:channel:1",
            get_sender_name=lambda: "tester",
            get_sender_id=lambda: "user-1",
        )

        allowed, reason = await plugin._passes_llm_safety_check(event, "普通消息")

        self.assertFalse(allowed)
        self.assertEqual(reason, "安全审核失败或超时")

    async def test_string_false_block_on_error_allows_provider_error(self):
        plugin = object.__new__(MsgTransfer)
        plugin.context = DummyContext()
        plugin.plugin_config = {
            "llm_safety_check": {
                "enabled": True,
                "block_on_error": "false",
                "timeout_seconds": 1,
            }
        }
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_id="discord-3"),
            unified_msg_origin="discord:channel:1",
            get_sender_name=lambda: "tester",
            get_sender_id=lambda: "user-1",
        )

        allowed, reason = await plugin._passes_llm_safety_check(event, "普通消息")

        self.assertTrue(allowed)
        self.assertEqual(reason, "安全审核失败或超时")

    async def test_string_false_enabled_disables_safety_check(self):
        plugin = object.__new__(MsgTransfer)
        plugin.context = DummyContext()
        plugin.plugin_config = {
            "llm_safety_check": {
                "enabled": "false",
                "block_on_error": True,
                "timeout_seconds": 1,
            }
        }
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_id="discord-4"),
            unified_msg_origin="discord:channel:1",
            get_sender_name=lambda: "tester",
            get_sender_id=lambda: "user-1",
        )

        allowed, reason = await plugin._passes_llm_safety_check(event, "普通消息")

        self.assertTrue(allowed)
        self.assertEqual(reason, "")

    async def test_empty_provider_list_uses_current_provider(self):
        provider = SuccessfulProvider()
        plugin = object.__new__(MsgTransfer)
        plugin.context = DummyContext(provider)
        plugin.plugin_config = {
            "llm_safety_check": {
                "enabled": True,
                "llm_providers": [],
                "system_prompt": "system",
            }
        }
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_id="discord-5"),
            unified_msg_origin="discord:channel:1",
            get_sender_name=lambda: "tester",
            get_sender_id=lambda: "user-1",
        )

        allowed, reason = await plugin._passes_llm_safety_check(event, "普通消息")

        self.assertTrue(allowed)
        self.assertEqual(reason, "内容正常")
        self.assertEqual(provider.calls[0]["system_prompt"], "system")
        self.assertEqual(provider.calls[0]["session_id"], "msg_transfer_safety:discord-5")

    async def test_astrbot_provider_template_uses_current_provider(self):
        provider = SuccessfulProvider('{"safe": false, "reason": "包含风险"}')
        plugin = object.__new__(MsgTransfer)
        plugin.context = DummyContext(provider)
        plugin.plugin_config = {
            "llm_safety_check": {
                "enabled": True,
                "llm_providers": [
                    {
                        "__template_key": "astrbot_provider",
                        "name": "当前 Provider",
                    }
                ],
            }
        }
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_id="discord-6"),
            unified_msg_origin="discord:channel:1",
            get_sender_name=lambda: "tester",
            get_sender_id=lambda: "user-1",
        )

        allowed, reason = await plugin._passes_llm_safety_check(event, "普通消息")

        self.assertFalse(allowed)
        self.assertEqual(reason, "包含风险")
        self.assertEqual(len(provider.calls), 1)

    async def test_provider_list_tries_next_provider_after_failure(self):
        provider = SuccessfulProvider()
        plugin = object.__new__(MsgTransfer)
        plugin.context = DummyContext(provider)
        plugin._call_openai_compatible_safety_provider = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        cfg = {
            "llm_providers": [
                {
                    "__template_key": "openai_compatible",
                    "name": "OpenAI",
                },
                {
                    "__template_key": "astrbot_provider",
                    "name": "AstrBot",
                },
            ],
            "system_prompt": "system",
            "timeout_seconds": 1,
        }

        result = await plugin._call_llm_safety(
            prompt="prompt",
            cfg=cfg,
            session_id="session-1",
            umo="discord:channel:1",
        )

        self.assertEqual(result, '{"safe": true, "reason": "内容正常"}')
        plugin._call_openai_compatible_safety_provider.assert_awaited_once()
        self.assertEqual(len(provider.calls), 1)

    async def test_responses_api_provider_builds_request_and_extracts_output_text(self):
        plugin = object.__new__(MsgTransfer)
        fake_client = MagicMock()
        fake_client.responses.create = AsyncMock(
            return_value=SimpleNamespace(
                output_text='{"safe": true, "reason": "Responses API 正常"}'
            )
        )
        fake_client.close = AsyncMock()
        provider = {
            "name": "Responses",
            "api_key": "test-key",
            "base_url": "https://api.openai.com",
            "model": "gpt-5",
        }
        cfg = {
            "reasoning_effort": "high",
            "timeout_seconds": 5,
        }

        with patch.object(module, "_AsyncOpenAI", return_value=fake_client) as client_factory:
            result = await plugin._call_responses_api_safety_provider(
                prompt="审核载荷",
                system_prompt="system",
                provider=provider,
                cfg=cfg,
            )

        self.assertEqual(result, '{"safe": true, "reason": "Responses API 正常"}')
        client_factory.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            timeout=5.0,
        )
        fake_client.responses.create.assert_awaited_once_with(
            model="gpt-5",
            instructions="system",
            input="审核载荷",
            max_output_tokens=512,
            reasoning={"effort": "high"},
        )
        fake_client.close.assert_awaited_once()

    async def test_responses_api_provider_accepts_base_url_with_v1_suffix(self):
        plugin = object.__new__(MsgTransfer)
        fake_client = MagicMock()
        fake_client.responses.create = AsyncMock(
            return_value=SimpleNamespace(output_text='{"safe": true}')
        )
        fake_client.close = AsyncMock()
        provider = {
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1/",
        }

        with patch.object(module, "_AsyncOpenAI", return_value=fake_client) as client_factory:
            await plugin._call_responses_api_safety_provider(
                prompt="prompt",
                system_prompt="system",
                provider=provider,
                cfg={},
            )

        client_factory.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            timeout=10.0,
        )

    def test_responses_api_provider_extracts_nested_output_text(self):
        response_json = {
            "output": [
                {"type": "reasoning", "content": []},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": '{"safe": false'},
                        {"type": "output_text", "text": ', "reason": "风险"}'},
                    ],
                },
            ]
        }

        result = MsgTransfer._extract_responses_api_text(response_json, "Responses")

        self.assertEqual(result, '{"safe": false, "reason": "风险"}')

    async def test_responses_api_template_uses_responses_provider(self):
        plugin = object.__new__(MsgTransfer)
        plugin._call_responses_api_safety_provider = AsyncMock(
            return_value='{"safe": true, "reason": "正常"}'
        )
        plugin._call_openai_compatible_safety_provider = AsyncMock(
            side_effect=AssertionError("Responses API 不应调用 Chat Completions")
        )
        cfg = {
            "llm_providers": [
                {
                    "__template_key": "responses_api",
                    "name": "Responses",
                }
            ],
            "system_prompt": "system",
            "timeout_seconds": 1,
        }

        result = await plugin._call_llm_safety(
            prompt="prompt",
            cfg=cfg,
            session_id="session-1",
            umo="discord:channel:1",
        )

        self.assertEqual(result, '{"safe": true, "reason": "正常"}')
        plugin._call_responses_api_safety_provider.assert_awaited_once()
        plugin._call_openai_compatible_safety_provider.assert_not_awaited()

    def test_provider_id_is_not_part_of_runtime_config(self):
        plugin = object.__new__(MsgTransfer)
        plugin.plugin_config = {
            "llm_safety_check": {
                "provider_id": "legacy-provider",
            }
        }

        config = plugin._get_llm_safety_config()

        self.assertNotIn("provider_id", config)
        self.assertIn("llm_providers", config)

    def test_schema_uses_provider_templates(self):
        schema = json.loads((REPO_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        safety_schema = schema["llm_safety_check"]

        self.assertNotIn("provider_id", safety_schema["items"])
        providers = safety_schema["items"]["llm_providers"]
        self.assertEqual(providers["type"], "template_list")
        self.assertEqual(
            set(providers["templates"]),
            {
                "openai_compatible",
                "responses_api",
                "astrbot_provider",
                "modelscope",
            },
        )
        # 验证新增的翻译配置段
        translation_schema = schema["llm_translation"]
        self.assertEqual(translation_schema["type"], "object")
        self.assertIn("enabled", translation_schema["items"])
        self.assertIn("llm_providers", translation_schema["items"])
        self.assertIn("timeout_seconds", translation_schema["items"])
        self.assertIn("system_prompt", translation_schema["items"])
        # 验证翻译供应商模板与安全筛查一致
        tl_providers = translation_schema["items"]["llm_providers"]
        self.assertEqual(tl_providers["type"], "template_list")
        self.assertEqual(set(tl_providers["templates"]), set(providers["templates"]))
        # 验证转发规则模板含 translation 字段
        forward_rules = schema["forward_rules"]
        self.assertEqual(forward_rules["type"], "template_list")
        self.assertEqual(set(forward_rules["templates"]), {"forward_rule"})
        self.assertIn("translation", forward_rules["templates"]["forward_rule"]["items"])

    def test_config_forward_rules_are_normalized(self):
        plugin = object.__new__(MsgTransfer)
        plugin.plugin_config = {
            "forward_rules": [
                {
                    "__template_key": "forward_rule",
                    "source_umo": "aiocqhttp:GroupMessage:123456",
                    "target_umo": "discord:ChannelMessage:987654",
                },
                {
                    "__template_key": "forward_rule",
                    "source_umo": "",
                    "target_umo": "discord:ChannelMessage:111111",
                },
                {
                    "__template_key": "forward_rule",
                    "source_umo": "aiocqhttp:GroupMessage:123456",
                    "target_umo": "discord:ChannelMessage:987654",
                },
            ]
        }

        rules = plugin._get_config_forward_rules()

        self.assertEqual(
            rules,
            {
                "config-1": {
                    "source_umo": "aiocqhttp:GroupMessage:123456",
                    "target_umo": "discord:ChannelMessage:987654",
                    "translation": {},
                }
            },
        )

    async def test_config_forward_rules_merge_with_stored_rules_without_duplicates(self):
        plugin = object.__new__(MsgTransfer)
        plugin.plugin_config = {
            "forward_rules": [
                {
                    "source_umo": "aiocqhttp:GroupMessage:123456",
                    "target_umo": "discord:ChannelMessage:987654",
                },
                {
                    "source_umo": "aiocqhttp:GroupMessage:123456",
                    "target_umo": "discord:ChannelMessage:111111",
                },
            ]
        }
        plugin.store = SimpleNamespace(
            list_rules=AsyncMock(
                return_value={
                    "1": {
                        "source_umo": "aiocqhttp:GroupMessage:123456",
                        "target_umo": "discord:ChannelMessage:987654",
                    },
                    "2": {
                        "source_umo": "aiocqhttp:GroupMessage:123456",
                        "target_umo": "qq:GroupMessage:222222",
                    },
                }
            )
        )

        rules = await plugin._list_forward_rules("aiocqhttp:GroupMessage:123456")

        self.assertEqual(
            rules,
            {
                "config-1": {
                    "source_umo": "aiocqhttp:GroupMessage:123456",
                    "target_umo": "discord:ChannelMessage:987654",
                    "translation": {},
                },
                "config-2": {
                    "source_umo": "aiocqhttp:GroupMessage:123456",
                    "target_umo": "discord:ChannelMessage:111111",
                    "translation": {},
                },
                "2": {
                    "source_umo": "aiocqhttp:GroupMessage:123456",
                    "target_umo": "qq:GroupMessage:222222",
                },
            },
        )


    # ------------------------------------------------------------------ #
    # 翻译功能测试
    # ------------------------------------------------------------------ #

    async def test_translation_disabled_when_top_level_off(self):
        plugin = object.__new__(MsgTransfer)
        plugin.plugin_config = {
            "llm_translation": {
                "enabled": False,
            }
        }
        event = SimpleNamespace(message_obj=SimpleNamespace(message_id="t1"))
        rule = {
            "source_umo": "discord:ChannelMessage:1",
            "target_umo": "aiocqhttp:GroupMessage:2",
            "translation": {"enabled": True, "target_language": "English"},
        }

        result = await plugin._translate_message(event, "你好", rule)

        self.assertIsNone(result)

    async def test_translation_disabled_when_rule_level_off(self):
        plugin = object.__new__(MsgTransfer)
        plugin.plugin_config = {
            "llm_translation": {
                "enabled": True,
                "system_prompt": "Translate into {target_language}: {source_text}",
            }
        }
        event = SimpleNamespace(message_obj=SimpleNamespace(message_id="t2"))
        rule = {
            "source_umo": "discord:ChannelMessage:1",
            "target_umo": "aiocqhttp:GroupMessage:2",
            "translation": {"enabled": False, "target_language": "English"},
        }

        result = await plugin._translate_message(event, "你好", rule)

        self.assertIsNone(result)

    async def test_translation_formats_prompt_correctly(self):
        plugin = object.__new__(MsgTransfer)
        plugin._call_llm = AsyncMock(return_value="Hello")
        plugin.plugin_config = {
            "llm_translation": {
                "enabled": True,
                "system_prompt": "Translate into {target_language}: {source_text}",
            }
        }
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_id="t3"),
            unified_msg_origin="discord:channel:1",
        )
        rule = {
            "source_umo": "discord:ChannelMessage:1",
            "target_umo": "aiocqhttp:GroupMessage:2",
            "translation": {"enabled": True, "target_language": "English"},
        }

        result = await plugin._translate_message(event, "你好", rule)

        self.assertEqual(result, "Hello")
        plugin._call_llm.assert_awaited_once()
        call_kwargs = plugin._call_llm.call_args[1]
        self.assertIn("你好", call_kwargs["prompt"])
        self.assertIn("English", call_kwargs["prompt"])
        self.assertEqual(call_kwargs["tag"], "翻译")
        # 验证 system_prompt 被清空（Hy-MT2 兼容）
        self.assertEqual(call_kwargs["cfg"]["system_prompt"], "")

    async def test_translation_formats_with_source_language(self):
        plugin = object.__new__(MsgTransfer)
        plugin._call_llm = AsyncMock(return_value="Hello")
        plugin.plugin_config = {
            "llm_translation": {
                "enabled": True,
                "system_prompt": "Translate {source_language} to {target_language}: {source_text}",
            }
        }
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_id="t4"),
            unified_msg_origin="discord:channel:1",
        )
        rule = {
            "source_umo": "discord:ChannelMessage:1",
            "target_umo": "aiocqhttp:GroupMessage:2",
            "translation": {
                "enabled": True,
                "target_language": "English",
                "source_language": "Chinese",
            },
        }

        result = await plugin._translate_message(event, "你好", rule)

        self.assertEqual(result, "Hello")
        prompt = plugin._call_llm.call_args[1]["prompt"]
        self.assertIn("Chinese", prompt)
        self.assertIn("English", prompt)
        self.assertIn("你好", prompt)

    async def test_translation_empty_text_returns_none(self):
        plugin = object.__new__(MsgTransfer)
        plugin.plugin_config = {
            "llm_translation": {
                "enabled": True,
            }
        }
        event = SimpleNamespace(message_obj=SimpleNamespace(message_id="t5"))
        rule = {
            "source_umo": "discord:ChannelMessage:1",
            "target_umo": "aiocqhttp:GroupMessage:2",
            "translation": {"enabled": True},
        }

        result = await plugin._translate_message(event, "", rule)

        self.assertIsNone(result)

    async def test_translation_calls_llm_and_returns_result(self):
        plugin = object.__new__(MsgTransfer)
        plugin._call_llm = AsyncMock(return_value="Translated text")
        plugin.plugin_config = {
            "llm_translation": {
                "enabled": True,
                "system_prompt": "Translate into {target_language}: {source_text}",
                "timeout_seconds": 15,
            }
        }
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_id="t6"),
            unified_msg_origin="discord:channel:1",
        )
        rule = {
            "source_umo": "discord:ChannelMessage:1",
            "target_umo": "aiocqhttp:GroupMessage:2",
            "translation": {"enabled": True, "target_language": "English"},
        }

        result = await plugin._translate_message(event, "原始文本", rule)

        self.assertEqual(result, "Translated text")
        plugin._call_llm.assert_awaited_once_with(
            prompt="Translate into English: 原始文本",
            cfg={
                "enabled": True,
                "system_prompt": "",
                "timeout_seconds": 15,
                "llm_providers": [],
            },
            session_id=plugin._build_translation_session_id(event),
            umo="discord:channel:1",
            tag="翻译",
        )

    async def test_translation_handles_llm_failure_gracefully(self):
        plugin = object.__new__(MsgTransfer)
        plugin._call_llm = AsyncMock(side_effect=asyncio.TimeoutError("超时"))
        plugin.plugin_config = {
            "llm_translation": {
                "enabled": True,
                "system_prompt": "Translate: {source_text}",
            }
        }
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_id="t7"),
            unified_msg_origin="discord:channel:1",
        )
        rule = {
            "source_umo": "discord:ChannelMessage:1",
            "target_umo": "aiocqhttp:GroupMessage:2",
            "translation": {"enabled": True, "target_language": "English"},
        }

        result = await plugin._translate_message(event, "text", rule)

        self.assertIsNone(result)

    def test_translation_config_parsed_from_rule(self):
        plugin = object.__new__(MsgTransfer)
        plugin.plugin_config = {
            "forward_rules": [
                {
                    "source_umo": "discord:ChannelMessage:111",
                    "target_umo": "aiocqhttp:GroupMessage:222",
                    "translation": {
                        "enabled": True,
                        "target_language": "日本語",
                        "source_language": "English",
                    },
                },
                {
                    "source_umo": "discord:ChannelMessage:333",
                    "target_umo": "qqofficial:GroupMessage:444",
                },
            ]
        }

        rules = plugin._get_config_forward_rules()

        self.assertEqual(
            rules["config-1"]["translation"],
            {
                "enabled": True,
                "target_language": "日本語",
                "source_language": "English",
            },
        )
        self.assertEqual(rules["config-2"]["translation"], {})

    def test_get_llm_translation_config_defaults(self):
        plugin = object.__new__(MsgTransfer)
        plugin.plugin_config = {}

        config = plugin._get_llm_translation_config()

        self.assertFalse(config["enabled"])
        self.assertEqual(config["llm_providers"], [])
        self.assertEqual(config["timeout_seconds"], 30)
        self.assertIn("{target_language}", config["system_prompt"])

    def test_get_llm_translation_config_merges_with_defaults(self):
        plugin = object.__new__(MsgTransfer)
        plugin.plugin_config = {
            "llm_translation": {
                "enabled": True,
                "timeout_seconds": 60,
            }
        }

        config = plugin._get_llm_translation_config()

        self.assertTrue(config["enabled"])
        self.assertEqual(config["timeout_seconds"], 60)
        self.assertEqual(config["llm_providers"], [])
        self.assertIn("{target_language}", config["system_prompt"])


if __name__ == "__main__":
    unittest.main()
