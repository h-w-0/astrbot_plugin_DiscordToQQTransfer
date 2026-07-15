import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
            {"openai_compatible", "astrbot_provider", "modelscope"},
        )
        forward_rules = schema["forward_rules"]
        self.assertEqual(forward_rules["type"], "template_list")
        self.assertEqual(set(forward_rules["templates"]), {"forward_rule"})

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
                },
                "config-2": {
                    "source_umo": "aiocqhttp:GroupMessage:123456",
                    "target_umo": "discord:ChannelMessage:111111",
                },
                "2": {
                    "source_umo": "aiocqhttp:GroupMessage:123456",
                    "target_umo": "qq:GroupMessage:222222",
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
