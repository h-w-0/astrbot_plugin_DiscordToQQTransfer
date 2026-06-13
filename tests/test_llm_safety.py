import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

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


class DummyContext:
    def get_using_provider(self):
        return DummyProvider()


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


if __name__ == "__main__":
    unittest.main()
