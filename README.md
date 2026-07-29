# MsgTransfer —— AstrBot 跨平台消息转发插件

> **本仓库基于 [Siaospeed/astrbot_plugin_msg_transfer](https://github.com/Siaospeed/astrbot_plugin_msg_transfer) 进行 fork 改动，遵守 AGPL-3.0 许可证。**

一个用于在 **QQ** 与 **Discord** 之间双向转发与同步消息的 AstrBot 插件，支持回复引用链还原、原生 @提及、图片转发等特性。

---

## ✨ 特性

- **QQ ↔ Discord 双向转发**：消息在两端之间自动同步。
- **消息来源标注**：自动在消息前标记发送者及来源平台（`[转发] 发送者名 (平台): 内容`）。
- **回复引用链还原**：
  - QQ 端回复已转发的消息时，自动引用原始 QQ 消息并 @ 原始发送者。
  - Discord 端引用回复时，自动还原为 Discord 原生引用 + 跳转链接。
- **Discord Webhook 集成**：自动为 Discord 频道创建 Webhook，以发送者身份显示头像和昵称。
- **原生 Discord @提及**：QQ → Discord 转发时，QQ @提及自动转换为 Discord 原生 `<@user_id>` 格式。
- **可持久化存储**：Webhook 与消息映射缓存自动保存，重启不丢失。

---

## 🚀 快速开始

1. **下载插件**：从本仓库的 Release 下载 `.zip` 文件，在 AstrBot WebUI 的插件页面中选择「从文件安装」。
2. **安装依赖**：重启 AstrBot 会自动安装依赖，也可手动执行：
    ```bash
    pip install -r requirements.txt
    ```
3. **（可选）Discord 原生 @提及**：如需 QQ→Discord 转发的 @提及功能，还需安装 `discord.py`：
    ```bash
    pip install discord.py>=2.0.0
    ```
4. **重启 AstrBot**。

---

## ⚙️ 配置

转发规则仅通过 Dashboard 的 `forward_rules` 配置。

- `forward_rules`：按 `source_umo → target_umo` 配置消息转发关系，支持动态增删。
- `content_safety.enabled`：仅对当前规则启用 LLM 内容审核；开启后不区分源或目标平台，均在发送前审核。
- 配置规则的 Discord 目标会在插件启动时自动尝试创建 Webhook。

`llm_safety_check` 仅提供所有规则共用的 LLM 审核配置：

- `llm_providers`：按顺序尝试的供应商列表，支持 `OpenAI 兼容`、`OpenAI Responses API`、`AstrBot 当前 Provider` 和 `ModelScope`。
- `llm_providers` 留空时使用当前会话的 AstrBot Provider。
- `OpenAI Responses API` 供应商默认请求 `https://api.openai.com/v1/responses`，也支持填写兼容 Responses API 的自定义地址。
- `block_on_error`：LLM 调用失败或超时时是否阻止转发。

`llm_translation` 提供所有翻译规则共用的 LLM 配置：

- `use_recent_context`：开启后，将同一来源会话最近 5 条原消息作为上下文提供给翻译模型，仅用于语义消歧；模型仍只输出当前消息的译文。该上下文仅保存在内存中，插件重启后清空。

---

## 📦 数据存储

插件在 `data/plugin_data/astrbot_plugin_DiscordToQQTransfer/` 下维护以下文件：

| 文件 | 用途 |
|------|------|
| `webhooks.json` | Discord Webhook URL 映射 |
| `mappings.json` | QQ 号 → QQ 昵称映射 |
| `msg_mapping.json` | QQ 消息 ID ↔ Discord 消息 ID 映射（含发送者信息） |
| `forward_log.json` | Discord 转发消息记录（用于多跳引用链还原） |

---

## 🔄 转发行为示例

假设已在 Dashboard 中配置 QQ 群 `654321` 与 Discord 频道 `123456` 的转发规则：

### QQ → Discord

```
QQ: mmyddd: 1
→ Discord (Webhook): mmyddd (QQ): 1
```

### Discord → QQ

```
Discord: mmyddd: 2
→ QQ: [转发] mmyddd (discord): 2
```

### 多跳引用链（QQ 回复 → Discord 引用回复 → ...）

```
① QQ: mmyddd: 1
② Discord (Webhook): mmyddd (QQ):  1
③ Discord: mmyddd (引用②): 2
④ QQ (引用①, @mmyddd): [转发] mmyddd (discord): 2
⑤ QQ: mmyddd (引用④): 3
⑥ Discord (引用③, @mmyddd): mmyddd (QQ): 3
```

---

## 🧩 项目结构

```
astrbot_plugin_msg_transfer/
├── LICENSE
├── README.md
├── main.py          # 插件主逻辑
├── webhook.py       # Discord Webhook 管理模块
├── metadata.yaml
└── requirements.txt
```

---

## 📜 许可证

- 本插件以 **AGPL-3.0** 开源。
- 基于 [Siaospeed/astrbot_plugin_msg_transfer](https://github.com/Siaospeed/astrbot_plugin_msg_transfer) 进行 fork 改动。
- 上游项目与 AstrBot 框架均基于 AGPL-3.0，因此本插件同样以 AGPL-3.0 分发。

---

## 🤝 贡献

欢迎提交 Issue 或 PR。
