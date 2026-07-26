import asyncio
import importlib.util
import json
import re
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

    async def test_same_source_messages_wait_for_prior_llm_work(self):
        plugin = object.__new__(MsgTransfer)
        plugin._list_forward_rules = AsyncMock(
            return_value={"config-1": {"target_umo": "aiocqhttp:GroupMessage:2"}}
        )
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        forwarded_ids = []

        async def slow_forward_rule(event, *_args):
            message_id = event.message_obj.message_id
            forwarded_ids.append(message_id)
            if message_id == "first":
                first_started.set()
                await release_first.wait()

        plugin._forward_single_rule = AsyncMock(side_effect=slow_forward_rule)

        def make_event(message_id: str):
            return SimpleNamespace(
                unified_msg_origin="discord:channel:1",
                message_obj=SimpleNamespace(
                    message_id=message_id,
                    raw_message={"post_type": "message"},
                ),
                get_messages=lambda: [SimpleNamespace(text=message_id)],
                get_platform_name=lambda: "aiocqhttp",
            )

        first_task = asyncio.create_task(plugin.forward_message(make_event("first")))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second_task = asyncio.create_task(plugin.forward_message(make_event("second")))
        await asyncio.sleep(0)

        self.assertEqual(forwarded_ids, ["first"])

        release_first.set()
        await asyncio.gather(first_task, second_task)

        self.assertEqual(forwarded_ids, ["first", "second"])

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

    async def test_legacy_global_enabled_is_ignored(self):
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

        self.assertFalse(allowed)
        self.assertEqual(reason, "安全审核失败或超时")

    def test_bundled_sensitive_lexicon_loads_vocabulary(self):
        lexicon = module.load_bundled_sensitive_lexicon()

        self.assertGreater(lexicon.word_count, 0)

    async def test_local_sensitive_lexicon_blocks_before_llm_call(self):
        plugin = object.__new__(MsgTransfer)
        plugin.plugin_config = {
            "llm_safety_check": {
                "本地词汇库增强过滤": True,
            }
        }
        plugin._call_llm_safety = AsyncMock()
        lexicon = MagicMock()
        lexicon.find_match.return_value = "匹配词"

        with patch.object(module, "load_bundled_sensitive_lexicon", return_value=lexicon):
            allowed, reason = await plugin._passes_llm_safety_check(
                SimpleNamespace(),
                "待检查消息",
            )

        self.assertFalse(allowed)
        self.assertEqual(reason, "命中本地词汇库：匹配词")
        lexicon.find_match.assert_called_once_with("待检查消息")
        plugin._call_llm_safety.assert_not_awaited()

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

    async def test_llm_response_is_logged_when_debug_switch_enabled(self):
        plugin = object.__new__(MsgTransfer)
        plugin.context = DummyContext(SuccessfulProvider("调试输出"))
        plugin.plugin_config = {"debug_log_llm_response": True}

        with patch.object(module.logger, "info") as log_info:
            result = await plugin._call_llm(
                prompt="测试提示词",
                cfg={"llm_providers": [], "timeout_seconds": 1},
                session_id="test-session",
                umo="discord:channel:1",
                tag="翻译",
            )

        self.assertEqual(result, "调试输出")
        log_info.assert_any_call("LLM 翻译返回内容: 调试输出")

    async def test_llm_response_is_not_logged_when_debug_switch_disabled(self):
        plugin = object.__new__(MsgTransfer)
        plugin.context = DummyContext(SuccessfulProvider("不应记录"))
        plugin.plugin_config = {"debug_log_llm_response": "false"}

        with patch.object(module.logger, "info") as log_info:
            await plugin._call_llm(
                prompt="测试提示词",
                cfg={"llm_providers": [], "timeout_seconds": 1},
                session_id="test-session",
                umo="discord:channel:1",
            )

        self.assertFalse(
            any("不应记录" in str(call) for call in log_info.call_args_list)
        )

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
            "llm_max_tokens": 2048,
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
            max_output_tokens=2048,
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
        self.assertEqual(next(iter(schema)), "forward_rules")
        self.assertEqual(schema["debug_log_llm_response"]["type"], "bool")
        self.assertFalse(schema["debug_log_llm_response"]["default"])
        safety_schema = schema["llm_safety_check"]

        self.assertNotIn("provider_id", safety_schema["items"])
        self.assertNotIn("enabled", safety_schema["items"])
        self.assertFalse(safety_schema["items"]["本地词汇库增强过滤"]["default"])
        self.assertEqual(safety_schema["items"]["本地词汇库增强过滤"]["type"], "bool")
        self.assertEqual(safety_schema["items"]["llm_max_tokens"]["default"], 512)
        self.assertIn("reasoning_effort", safety_schema["items"])
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
        self.assertEqual(translation_schema["items"]["llm_max_tokens"]["default"], 512)
        self.assertEqual(translation_schema["items"]["reasoning_effort"]["default"], "")
        self.assertIn("system_prompt", translation_schema["items"])
        # 验证翻译供应商模板与安全筛查一致
        tl_providers = translation_schema["items"]["llm_providers"]
        self.assertEqual(tl_providers["type"], "template_list")
        self.assertEqual(set(tl_providers["templates"]), set(providers["templates"]))
        # 验证转发规则模板含翻译和内容审核字段
        forward_rules = schema["forward_rules"]
        self.assertEqual(forward_rules["type"], "template_list")
        self.assertEqual(set(forward_rules["templates"]), {"forward_rule"})
        self.assertIn("translation", forward_rules["templates"]["forward_rule"]["items"])
        content_safety = forward_rules["templates"]["forward_rule"]["items"]["content_safety"]
        self.assertEqual(content_safety["items"]["enabled"]["default"], False)

    def test_command_binding_handlers_are_removed(self):
        for handler_name in ("mt", "cmd_add", "cmd_bind", "cmd_del", "cmd_list"):
            with self.subTest(handler_name=handler_name):
                self.assertFalse(hasattr(MsgTransfer, handler_name))

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
                    "content_safety": {"enabled": False},
                }
            },
        )

    async def test_config_forward_rules_are_the_only_rule_source(self):
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
                side_effect=AssertionError("不应读取持久化转发规则")
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
                    "content_safety": {"enabled": False},
                },
                "config-2": {
                    "source_umo": "aiocqhttp:GroupMessage:123456",
                    "target_umo": "discord:ChannelMessage:111111",
                    "translation": {},
                    "content_safety": {"enabled": False},
                },
            },
        )
        plugin.store.list_rules.assert_not_awaited()

    def test_safety_payload_is_platform_neutral(self):
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_id="message-1"),
            unified_msg_origin="aiocqhttp:GroupMessage:123456",
            get_sender_name=lambda: "tester",
            get_sender_id=lambda: "user-1",
        )

        payload = json.loads(
            MsgTransfer._build_llm_safety_payload(
                event,
                "普通消息",
                "discord:ChannelMessage:987654",
            )
        )

        self.assertEqual(payload["task"], "audit_message_for_forwarding")
        self.assertNotIn("discord_message", payload)
        self.assertEqual(payload["forwarding_message"]["source_umo"], event.unified_msg_origin)
        self.assertEqual(payload["forwarding_message"]["target_umo"], "discord:ChannelMessage:987654")

    async def test_rule_content_safety_applies_to_webhook_forwarding(self):
        plugin = object.__new__(MsgTransfer)
        plugin.store = SimpleNamespace(
            update_mapping=AsyncMock(return_value=False),
            get_webhook_url=AsyncMock(return_value="https://example.invalid/webhook"),
        )
        plugin._passes_llm_safety_check = AsyncMock(return_value=(True, ""))
        plugin._forward_with_webhook = AsyncMock(return_value=True)
        event = SimpleNamespace(
            get_platform_name=lambda: "aiocqhttp",
            get_sender_id=lambda: "123456",
            get_sender_name=lambda: "tester",
        )
        rule = {
            "target_umo": "discord:ChannelMessage:987654",
            "content_safety": {"enabled": True},
        }

        await plugin._forward_single_rule(
            event,
            rule,
            "config-1",
            "aiocqhttp:GroupMessage:123456",
            [],
        )

        plugin._passes_llm_safety_check.assert_awaited_once_with(
            event,
            "",
            "discord:ChannelMessage:987654",
        )
        plugin._forward_with_webhook.assert_awaited_once_with(
            event,
            "discord:ChannelMessage:987654",
            [],
            "config-1",
            "https://example.invalid/webhook",
            rule,
        )

    async def test_rule_content_safety_blocks_before_any_send_path(self):
        plugin = object.__new__(MsgTransfer)
        plugin.store = SimpleNamespace(
            get_webhook_url=AsyncMock(),
        )
        plugin._passes_llm_safety_check = AsyncMock(return_value=(False, "包含风险"))
        plugin._reply_safety_block = AsyncMock()
        plugin._forward_with_webhook = AsyncMock()
        event = SimpleNamespace(
            get_platform_name=lambda: "discord",
        )
        rule = {
            "target_umo": "aiocqhttp:GroupMessage:987654",
            "content_safety": {"enabled": True},
        }

        await plugin._forward_single_rule(
            event,
            rule,
            "config-1",
            "discord:ChannelMessage:123456",
            [],
        )

        plugin._passes_llm_safety_check.assert_awaited_once_with(
            event,
            "",
            "aiocqhttp:GroupMessage:987654",
        )
        plugin._reply_safety_block.assert_awaited_once_with(
            event,
            "aiocqhttp:GroupMessage:987654",
            "包含风险",
        )
        plugin.store.get_webhook_url.assert_not_awaited()
        plugin._forward_with_webhook.assert_not_awaited()

    async def test_rule_content_safety_disabled_skips_check(self):
        plugin = object.__new__(MsgTransfer)
        plugin.store = SimpleNamespace(
            get_webhook_url=AsyncMock(return_value="https://example.invalid/webhook"),
        )
        plugin._passes_llm_safety_check = AsyncMock()
        plugin._forward_with_webhook = AsyncMock(return_value=True)
        event = SimpleNamespace(
            get_platform_name=lambda: "discord",
        )
        rule = {
            "target_umo": "discord:ChannelMessage:987654",
            "content_safety": {"enabled": False},
        }

        await plugin._forward_single_rule(
            event,
            rule,
            "config-1",
            "discord:ChannelMessage:123456",
            [],
        )

        plugin._passes_llm_safety_check.assert_not_awaited()
        plugin._forward_with_webhook.assert_awaited_once()

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

        # langdetect 检测"你好"的源语言，验证英文目标前缀格式正确
        detected = plugin._detect_source_language("你好")
        expected = f"(Translated from {detected})Hello"
        self.assertEqual(result, expected)
        plugin._call_llm.assert_awaited_once()
        call_kwargs = plugin._call_llm.call_args[1]
        self.assertIn("你好", call_kwargs["prompt"])
        self.assertIn("English", call_kwargs["prompt"])
        self.assertNotIn("{source_text}", call_kwargs["prompt"])
        self.assertNotIn("{target_language}", call_kwargs["prompt"])
        self.assertEqual(call_kwargs["tag"], "翻译")
        # 验证 system_prompt 被清空（Hy-MT2 兼容）
        self.assertEqual(call_kwargs["cfg"]["system_prompt"], "")

    def test_translation_prefix_uses_target_language(self):
        self.assertEqual(
            MsgTransfer._format_translation_prefix("Chinese", "English"),
            "(Translated from Chinese)",
        )
        self.assertEqual(
            MsgTransfer._format_translation_prefix("English", "中文"),
            "(从英文翻译)",
        )
        self.assertEqual(
            MsgTransfer._format_translation_prefix("Chinese", "en"),
            "(Translated from Chinese)",
        )

    async def test_translation_detects_source_language(self):
        """验证自动检测源语言并显示正确的英文前缀"""
        plugin = object.__new__(MsgTransfer)
        plugin._call_llm = AsyncMock(return_value="Bonjour le monde")
        plugin.plugin_config = {
            "llm_translation": {
                "enabled": True,
                "system_prompt": "Translate into {target_language}: {source_text}",
            }
        }
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_id="t4"),
            unified_msg_origin="discord:channel:1",
        )
        rule = {
            "source_umo": "discord:ChannelMessage:1",
            "target_umo": "aiocqhttp:GroupMessage:2",
            "translation": {"enabled": True, "target_language": "French"},
        }

        # langdetect 检测源语言，验证英文目标前缀格式正确
        result = await plugin._translate_message(event, "Hello world", rule)
        m = re.match(r'^\(Translated from (.+?)\)', result)
        self.assertIsNotNone(m, f"结果 '{result}' 应匹配英文前缀格式")
        self.assertIsNotNone(m.group(1), "应检测到源语言")
        self.assertIn("Bonjour le monde", result)

        # 验证 source_language 占位符被正确替换
        prompt = plugin._call_llm.call_args[1]["prompt"]
        self.assertNotIn("{source_language}", prompt)

    def test_source_language_detection_prefers_distinctive_scripts(self):
        """短 CJK 文本使用文字系统判断，避免 langdetect 误判"""
        cases = {
            "不知道": "Chinese",
            "原始文本": "Chinese",
            "繁體中文": "Chinese",
            "你好 hello": "Chinese",
            "日本語です": "Japanese",
            "おはよう": "Japanese",
            "한글입니다": "Korean",
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(MsgTransfer._detect_source_language(text), expected)

    def test_source_language_detection_falls_back_to_langdetect(self):
        """没有明确文字系统时继续使用 langdetect"""
        with patch.object(module, "_detect_lang", return_value="fr") as detect_lang:
            result = MsgTransfer._detect_source_language("Bonjour tout le monde")

        self.assertEqual(result, "French")
        detect_lang.assert_called_once_with("Bonjour tout le monde")

    async def test_translation_uses_rule_source_language_override(self):
        """验证规则中手动指定 source_language 时优先使用"""
        plugin = object.__new__(MsgTransfer)
        plugin._call_llm = AsyncMock(return_value="Hola")
        plugin.plugin_config = {
            "llm_translation": {
                "enabled": True,
                "system_prompt": "Translate from {source_language} to {target_language}: {source_text}",
            }
        }
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_id="t5"),
            unified_msg_origin="discord:channel:1",
        )
        rule = {
            "source_umo": "discord:ChannelMessage:1",
            "target_umo": "aiocqhttp:GroupMessage:2",
            "translation": {
                "enabled": True,
                "target_language": "Chinese",
                "source_language": "English",
            },
        }

        result = await plugin._translate_message(event, "Hello", rule)

        self.assertEqual(result, "(从英文翻译)Hola")
        prompt = plugin._call_llm.call_args[1]["prompt"]
        self.assertIn("Translate from English to Chinese: Hello", prompt)

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

        # langdetect 检测 "原始文本" 的实际结果（可能因环境不同略有差异）
        self.assertIn("Translated from Chinese)Translated text", result)
        plugin._call_llm.assert_awaited_once_with(
            prompt="Translate into English: 原始文本",
            cfg={
                "enabled": True,
                "system_prompt": "",
                "timeout_seconds": 15,
                "llm_max_tokens": 512,
                "reasoning_effort": "",
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
        self.assertEqual(config["llm_max_tokens"], 512)
        self.assertEqual(config["reasoning_effort"], "")
        self.assertIn("{target_language}", config["system_prompt"])

    def test_get_llm_translation_config_merges_with_defaults(self):
        plugin = object.__new__(MsgTransfer)
        plugin.plugin_config = {
            "llm_translation": {
                "enabled": True,
                "timeout_seconds": 60,
                "llm_max_tokens": 2048,
                "reasoning_effort": "high",
            }
        }

        config = plugin._get_llm_translation_config()

        self.assertTrue(config["enabled"])
        self.assertEqual(config["timeout_seconds"], 60)
        self.assertEqual(config["llm_max_tokens"], 2048)
        self.assertEqual(config["reasoning_effort"], "high")
        self.assertEqual(config["llm_providers"], [])
        self.assertIn("{target_language}", config["system_prompt"])


if __name__ == "__main__":
    unittest.main()
