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


def make_event(message_id="merge-1"):
    return SimpleNamespace(
        message_obj=SimpleNamespace(message_id=message_id),
        get_sender_name=lambda: "发起人",
        get_sender_id=lambda: "42",
        get_platform_name=lambda: "aiocqhttp",
        get_self_id=lambda: "bot",
        unified_msg_origin="aiocqhttp:GroupMessage:123",
    )


def make_plugin():
    plugin = object.__new__(MsgTransfer)
    plugin.plugin_config = {}
    return plugin


class MergedForwardTests(IsolatedAsyncioTestCase):
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
