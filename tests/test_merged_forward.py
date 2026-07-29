import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

try:
    from .test_llm_safety import MsgTransfer
except ImportError:
    from test_llm_safety import MsgTransfer


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakePlain:
    type = "Plain"

    def __init__(self, text):
        self.text = text


class FakeNode:
    type = "Node"

    def __init__(self, name, content, uin="100"):
        self.name = name
        self.uin = uin
        self.content = content


class FakeNodes:
    type = "Nodes"

    def __init__(self, nodes):
        self.nodes = nodes


class FakeForward:
    type = "Forward"

    def __init__(self, forward_id, content=None):
        self.id = forward_id
        if content is not None:
            self.content = content


def make_event(message_id="merge-1", bot=None):
    event = SimpleNamespace(
        message_obj=SimpleNamespace(message_id=message_id),
        get_sender_name=lambda: "发起人",
        get_sender_id=lambda: "42",
        get_platform_name=lambda: "aiocqhttp",
        get_self_id=lambda: "bot",
        unified_msg_origin="aiocqhttp:GroupMessage:123",
    )
    if bot is not None:
        event.bot = bot
    return event


def make_plugin():
    plugin = object.__new__(MsgTransfer)
    plugin.plugin_config = {}
    return plugin


class MergedForwardTests(IsolatedAsyncioTestCase):
    async def test_inline_forward_nodes_are_used_before_remote_lookup(self):
        async def call_action(*_args, **_kwargs):
            raise AssertionError("inline content should not call OneBot")

        plugin = make_plugin()
        event = make_event(
            bot=SimpleNamespace(api=SimpleNamespace(call_action=call_action))
        )
        resolved = await plugin._resolve_merged_forward_message(
            event,
            [
                FakeForward(
                    "inline",
                    content=[
                        {
                            "sender": {"nickname": "Alice", "user_id": "1"},
                            "message": [
                                {"type": "text", "data": {"text": "inline"}}
                            ],
                        }
                    ],
                )
            ],
        )

        self.assertIn("inline", plugin._format_merged_forward_text(resolved))

    async def test_inner_forward_id_falls_back_to_outer_event_message_id(self):
        calls = []

        async def call_action(action, **params):
            calls.append((action, params))
            if action == "get_forward_msg" and params.get("message_id") == "inner-1":
                return {
                    "status": "failed",
                    "retcode": 1200,
                    "data": None,
                    "message": "消息已过期或者为内层消息",
                }
            if action == "get_forward_msg" and params.get("message_id") == "outer-1":
                return {
                    "data": {
                        "messages": [
                            {
                                "sender": {"nickname": "Alice", "user_id": "1"},
                                "message": [
                                    {"type": "text", "data": {"text": "restored"}}
                                ],
                            }
                        ]
                    }
                }
            raise RuntimeError("message is unavailable")

        plugin = make_plugin()
        event = make_event(
            message_id="outer-1",
            bot=SimpleNamespace(api=SimpleNamespace(call_action=call_action)),
        )
        resolved = await plugin._resolve_merged_forward_message(
            event,
            [FakeForward("inner-1")],
        )

        rendered = plugin._format_merged_forward_text(resolved)

        self.assertIn("restored", rendered)
        self.assertNotIn("合并转发解析失败", rendered)
        self.assertIn(("get_forward_msg", {"message_id": "inner-1"}), calls)
        self.assertIn(("get_forward_msg", {"message_id": "outer-1"}), calls)

    async def test_forward_component_is_fetched_and_nested_records_keep_order(self):
        calls = []
        payloads = {
            "root": {
                "data": {
                    "messages": [
                        {
                            "sender": {"nickname": "Alice", "user_id": "1"},
                            "message": [
                                {"type": "text", "data": {"text": "first"}},
                                {"type": "forward", "data": {"id": "nested"}},
                                {"type": "text", "data": {"text": "after"}},
                            ],
                        },
                        {
                            "sender": {"nickname": "Carol", "user_id": "3"},
                            "message": [
                                {"type": "text", "data": {"text": "second"}},
                            ],
                        },
                    ]
                }
            },
            "nested": {
                "data": {
                    "messages": [
                        {
                            "sender": {"nickname": "Bob", "user_id": "2"},
                            "message": [
                                {"type": "text", "data": {"text": "nested"}},
                            ],
                        }
                    ]
                }
            },
        }

        async def call_action(_action, **params):
            calls.append(params)
            return payloads[str(params.get("message_id") or params.get("id"))]

        plugin = make_plugin()
        event = make_event(
            bot=SimpleNamespace(api=SimpleNamespace(call_action=call_action))
        )
        resolved = await plugin._resolve_merged_forward_message(
            event,
            [FakeForward("root")],
        )

        units = plugin._build_merged_forward_units(resolved)
        rendered = plugin._format_merged_forward_text(resolved)

        self.assertEqual([unit["path"] for unit in units], [(1,), (1, 1), (1,), (2,)])
        self.assertEqual(
            [
                unit["components"][0].get("data", {}).get("text")
                for unit in units
                if unit["components"]
            ],
            ["first", "nested", "after", "second"],
        )
        self.assertLess(rendered.index("first"), rendered.index("nested"))
        self.assertLess(rendered.index("nested"), rendered.index("after"))
        self.assertLess(rendered.index("after"), rendered.index("second"))
        self.assertEqual(
            [str(params.get("message_id")) for params in calls],
            ["root", "nested"],
        )

    async def test_forward_fetch_failure_keeps_readable_placeholder(self):
        async def call_action(_action, **_params):
            raise RuntimeError("OneBot unavailable")

        plugin = make_plugin()
        event = make_event(
            bot=SimpleNamespace(api=SimpleNamespace(call_action=call_action))
        )
        resolved = await plugin._resolve_merged_forward_message(
            event,
            [FakeForward("broken")],
        )

        rendered = plugin._format_merged_forward_text(resolved)

        self.assertIn("[合并转发解析失败: broken]", rendered)

    async def test_forward_without_bot_does_not_raise(self):
        plugin = make_plugin()

        resolved = await plugin._resolve_merged_forward_message(
            make_event(),
            [FakeForward("missing-bot")],
        )

        self.assertIn(
            "[合并转发解析失败: missing-bot]",
            plugin._format_merged_forward_text(resolved),
        )

    def test_nested_nodes_keep_order_and_hierarchy(self):
        plugin = make_plugin()
        message_chain = [
            FakeNodes([
                FakeNode(
                    "Alice",
                    [
                        FakePlain("one"),
                        FakeNodes([FakeNode("Bob", [FakePlain("nested")])]),
                        FakePlain("after"),
                    ],
                ),
                FakeNode("Carol", [FakePlain("two")]),
            ])
        ]

        units = plugin._build_merged_forward_units(message_chain)

        self.assertEqual([unit["path"] for unit in units], [(1,), (1, 1), (1,), (2,)])
        self.assertTrue(units[0]["components"][0].text == "one")
        self.assertTrue(units[1]["components"][0].text == "nested")
        self.assertTrue(units[2]["continuation"])
        self.assertTrue(units[3]["components"][0].text == "two")
        rendered = plugin._format_merged_forward_text(message_chain)
        self.assertLess(rendered.index("one"), rendered.index("nested"))
        self.assertLess(rendered.index("nested"), rendered.index("after"))
        self.assertLess(rendered.index("after"), rendered.index("two"))

    async def test_translation_is_only_used_when_record_option_is_enabled(self):
        plugin = make_plugin()
        plugin.plugin_config = {"llm_translation": {"enabled": True}}
        plugin._translate_message = AsyncMock(return_value="translated")
        event = make_event()
        node = FakeNode("Alice", [FakePlain("original")])

        content, _, _ = await plugin._prepare_merged_forward_unit(
            event,
            {"path": (1,), "depth": 0, "sender": "Alice", "components": node.content},
            {
                "translation": {
                    "enabled": True,
                    "translate_forward_records": False,
                }
            },
            {},
        )

        self.assertIn("original", content)
        self.assertNotIn("translated", content)
        plugin._translate_message.assert_not_awaited()

        content, _, _ = await plugin._prepare_merged_forward_unit(
            event,
            {"path": (1,), "depth": 0, "sender": "Alice", "components": node.content},
            {
                "translation": {
                    "enabled": True,
                    "translate_forward_records": True,
                }
            },
            {},
        )

        self.assertIn("translated", content)
        plugin._translate_message.assert_awaited_once()

    async def test_translation_failure_falls_back_to_original_text(self):
        plugin = make_plugin()
        plugin.plugin_config = {"llm_translation": {"enabled": True}}
        plugin._translate_message = AsyncMock(side_effect=asyncio.TimeoutError("timeout"))

        content, _, _ = await plugin._prepare_merged_forward_unit(
            make_event(),
            {"path": (1,), "depth": 0, "sender": "Alice", "components": [FakePlain("original")]},
            {
                "translation": {
                    "enabled": True,
                    "translate_forward_records": True,
                }
            },
            {},
        )

        self.assertIn("original", content)
        self.assertNotIn("None", content)
        plugin._translate_message.assert_awaited_once()

    async def test_translation_localizes_forward_headers_without_translating_usernames(self):
        plugin = make_plugin()
        plugin.plugin_config = {"llm_translation": {"enabled": True}}
        plugin._translate_message = AsyncMock(return_value="translated")
        rule = {
            "translation": {
                "enabled": True,
                "target_language": "English",
                "translate_forward_records": True,
            }
        }

        root_content, _, _ = await plugin._prepare_merged_forward_unit(
            make_event(),
            {
                "path": (1,),
                "depth": 0,
                "sender": "新月",
                "components": [
                    FakePlain("original"),
                    {"type": "mface", "data": {}},
                ],
            },
            rule,
            {},
        )
        nested_content, _, _ = await plugin._prepare_merged_forward_unit(
            make_event(),
            {
                "path": (1, 1),
                "depth": 1,
                "sender": "Muddy",
                "components": [FakePlain("nested")],
            },
            rule,
            {},
        )

        self.assertIn("Forward Record 1", root_content)
        self.assertIn("Nested Forward 1.1", nested_content)
        self.assertIn("新月 (QQ)", root_content)
        self.assertIn("Muddy (QQ)", nested_content)
        self.assertIn("[Emoji]", root_content)
        self.assertNotIn("转发记录", root_content)
        self.assertNotIn("嵌套转发", nested_content)
        self.assertEqual(plugin._translate_message.await_count, 2)

    async def test_thread_creation_and_ordered_sending(self):
        plugin = make_plugin()
        plugin.webhook_manager = SimpleNamespace(
            create_thread_for_channel=AsyncMock(return_value=SimpleNamespace(id=987)),
            send_webhook_message=AsyncMock(side_effect=["discord-1", "discord-2"]),
        )
        plugin.store = SimpleNamespace(
            load_mappings=AsyncMock(return_value={}),
            set_msg_mapping=AsyncMock(),
        )
        message_chain = [FakeNodes([
            FakeNode("Alice", [FakePlain("first")]),
            FakeNode("Bob", [FakePlain("second")]),
        ])]

        result = await plugin._forward_merged_forward_with_webhook(
            make_event(),
            "discord:ChannelMessage:123456",
            message_chain,
            "config-1",
            "https://example.invalid/webhook",
            {},
        )

        self.assertTrue(result)
        plugin.webhook_manager.create_thread_for_channel.assert_awaited_once()
        self.assertEqual(
            plugin.webhook_manager.create_thread_for_channel.await_args.args,
            (123456, "合并转发 - 发起人 - merge-1"),
        )
        self.assertEqual(plugin.webhook_manager.send_webhook_message.await_count, 2)
        calls = plugin.webhook_manager.send_webhook_message.await_args_list
        self.assertEqual([call.kwargs["thread_id"] for call in calls], ["987", "987"])
        self.assertEqual(
            [call.kwargs["content"].splitlines()[-1] for call in calls],
            ["first", "second"],
        )
        plugin.store.set_msg_mapping.assert_awaited_once()

    async def test_thread_title_uses_translation_language_and_preserves_sender_name(self):
        plugin = make_plugin()
        plugin.plugin_config = {"llm_translation": {"enabled": True}}
        plugin._translate_message = AsyncMock(return_value="translated")
        plugin.webhook_manager = SimpleNamespace(
            create_thread_for_channel=AsyncMock(return_value=SimpleNamespace(id=987)),
            send_webhook_message=AsyncMock(return_value="discord-1"),
        )
        plugin.store = SimpleNamespace(
            load_mappings=AsyncMock(return_value={}),
            set_msg_mapping=AsyncMock(),
        )
        rule = {
            "translation": {
                "enabled": True,
                "target_language": "English",
                "translate_forward_records": True,
            }
        }

        result = await plugin._forward_merged_forward_with_webhook(
            make_event(),
            "discord:ChannelMessage:123456",
            [FakeNodes([FakeNode("Alice", [FakePlain("first")])])],
            "config-1",
            "https://example.invalid/webhook",
            rule,
        )

        self.assertTrue(result)
        self.assertEqual(
            plugin.webhook_manager.create_thread_for_channel.await_args.args,
            (123456, "Merged Forward - 发起人 - merge-1"),
        )

    async def test_thread_creation_or_single_send_failure_does_not_raise(self):
        plugin = make_plugin()
        plugin.store = SimpleNamespace(
            load_mappings=AsyncMock(return_value={}),
            set_msg_mapping=AsyncMock(),
        )
        plugin.webhook_manager = SimpleNamespace(
            create_thread_for_channel=AsyncMock(return_value=None),
            send_webhook_message=AsyncMock(),
        )
        message_chain = [FakeNodes([FakeNode("Alice", [FakePlain("first")])])]

        result = await plugin._forward_merged_forward_with_webhook(
            make_event(),
            "discord:ChannelMessage:123456",
            message_chain,
            "config-1",
            "https://example.invalid/webhook",
            {},
        )

        self.assertFalse(result)
        plugin.webhook_manager.send_webhook_message.assert_not_awaited()

        plugin.webhook_manager.create_thread_for_channel.return_value = SimpleNamespace(id=987)
        plugin.webhook_manager.send_webhook_message = AsyncMock(side_effect=[None, "discord-2"])
        message_chain = [FakeNodes([
            FakeNode("Alice", [FakePlain("first")]),
            FakeNode("Bob", [FakePlain("second")]),
        ])]

        result = await plugin._forward_merged_forward_with_webhook(
            make_event("merge-2"),
            "discord:ChannelMessage:123456",
            message_chain,
            "config-1",
            "https://example.invalid/webhook",
            {},
        )

        self.assertFalse(result)
        self.assertEqual(plugin.webhook_manager.send_webhook_message.await_count, 2)

    async def test_unknown_component_is_visible_as_placeholder(self):
        class Unsupported:
            pass

        plugin = make_plugin()
        content, _, _ = await plugin._prepare_merged_forward_unit(
            make_event(),
            {"path": (1,), "depth": 0, "sender": "Alice", "components": [Unsupported()]},
            {},
            {},
        )

        self.assertIn("不支持的消息类型: Unsupported", content)

    async def test_qq_super_face_dict_is_rendered_as_placeholder(self):
        plugin = make_plugin()
        content, _, _ = await plugin._prepare_merged_forward_unit(
            make_event(),
            {
                "path": (1,),
                "depth": 0,
                "sender": "Alice",
                "components": [
                    {"type": "mface", "data": {"summary": "超级表情"}},
                ],
            },
            {},
            {},
        )

        self.assertIn("[表情]", content)
        self.assertNotIn("不支持的消息类型", content)

    async def test_qq_super_face_object_is_rendered_as_placeholder(self):
        class SuperFace:
            type = "mface"

        plugin = make_plugin()
        content, _, _ = await plugin._prepare_merged_forward_unit(
            make_event(),
            {
                "path": (1,),
                "depth": 0,
                "sender": "Alice",
                "components": [SuperFace()],
            },
            {},
            {},
        )

        self.assertIn("[表情]", content)
        self.assertNotIn("不支持的消息类型", content)

    def test_schema_contains_safe_default_for_record_translation(self):
        schema = json.loads((REPO_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        translation = schema["forward_rules"]["templates"]["forward_rule"]["items"]["translation"]["items"]
        self.assertEqual(translation["translate_forward_records"]["type"], "bool")
        self.assertFalse(translation["translate_forward_records"]["default"])

    def test_long_content_is_split_without_losing_characters(self):
        content = "a" * 1995 + "\n" + "b" * 30

        chunks = MsgTransfer._split_discord_content(content)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 2000 for chunk in chunks))
        self.assertEqual("".join(chunks), content)


if __name__ == "__main__":
    import unittest

    unittest.main()
