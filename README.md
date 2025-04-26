# 三种方法实现监督微调 (SFT)：LLaMA Factory, trl 和 unsloth

> 我尝试了目前流行的三种微调框架，最推荐 unsloth，因为它快！LLaMA Factory 和 trl 是在夜里跑的，显卡风扇响了一宿。第二天看日志时，发现它们跑了三个小时才跑完。但是 unsloth 五分钟就跑完了，快得都有些夸张了。当然，这么比较不是完全公平的，因为它们的量化方法、LoRA 参数都是不同的。但是 unsloth 快这一点依然是非常显著的。如果你想在 GPU 服务器上认真微调，那么用 LLaMA Factory 没毛病。但如果只是在笔记本上随便玩玩，unsloth 的优势就太明显了！

## 一、引言

大语言模型有很强的通用能力，但在特定领域，它的表现不如领域小模型。为了让大模型适应特定任务，我们对大模型进行微调，使大模型在保持通用性的同时，兼具领域模型的专业知识、对话风格和输出格式等特质。

微调大模型有三种范式：

1. **无监督微调**：在海量数据上进行二次预训练
   - PT 增量预训练
2. **监督微调 (SFT)**：构造领域数据集，增强模型的指令遵循能力，并注入领域知识
   - 指令微调
3. **强化学习微调**：通过 reward 引导模型优化
   - [RLHF](https://arxiv.org/abs/2203.02155) 基于人类反馈的强化学习
   - [DPO](https://arxiv.org/abs/2305.18290) 直接偏好优化方法
   - [ORPO](https://arxiv.org/abs/2403.07691) 比值比偏好优化
   - [GRPO](https://arxiv.org/abs/2402.03300) 群体相对策略优化

本文聚焦 **监督微调 (Supervised Fine-Tuning)**。监督微调是一种简单但有效的微调方式，能够快速融合业务数据、适应业务场景，因此它的性价比极高！

### 1. SFT 的简单介绍

监督微调的优化目标是 **最小化模型生成回答与目标回答之间的差异**，通常使用交叉熵损失。为避免破坏预训练阶段获得的知识，SFT 阶段通常使用 **较低的学习率**，并且只更新部分参数层，其他参数保持不变。与预训练阶段所需的海量数据相比，SFT 只需 **较小的数据量**（数千到数十万样本），即可完成微调。

### 2. SFT 的使用场景

为了让大家感受一下 SFT 能做什么，下面列举一些使用场景：

|任务|场景举例|类型|
| -- | -- | -- |
|**文案生成**|输入标签：`红色#女士#卫衣`，输出文案：`女士专属红色卫衣，解锁秋冬时尚密码`|任务对齐|
|**情感分类**|输入用户评论：`蓝牙连接不稳定`，输出情感标签：`负面`|任务对齐|
|**合同审核**|输入合同文本，输出潜在法律风险，并引述法条和案例|知识迁移|

### 3. SFT 的数据集格式

SFT 通过人类精心设计的高质量数据集进行微调。

微调使用的数据格式是灵活的。但是过于灵活的数据格式，可能导致加载数据的不便。SFT 经过几年时间，也逐渐发展出一些主流的数据格式。其中，[alpaca](https://github.com/tatsu-lab/stanford_alpaca) 就是一种专为指令微调设计的数据格式。通常，每条 alpaca 数据由 `instruction`, `input`, `output` 三个字段组成。

**1）问答数据集**

```
{
    "instruction": "帕金森叠加综合征的辅助治疗有些什么？",
    "input": "",
    "output": "综合治疗；康复训练；生活护理指导；低频重复经颅磁刺激治疗"
}
```

上面是一条 alpaca 格式的问答数据。对于问答数据，`input` 字段可以留空。问题放在 `instruction` 字段；回答放在 `output` 字段。

**2）指令微调数据集**

```
{
    "instruction": "请对下面这篇文章进行分类，分类标签从“教育”、“健康”、“游戏”、“其他”四个标签中选择。仅回答标签，不要回答除标签以外的任何内容。",
    "input": "怪物猎人崛起实在是太好玩了！",
    "output": "游戏"
}
```

上面是一条 alpaca 格式的指令微调数据。指令微调数据的三个字段都有值。`instruction` 字段写我们希望模型做什么；`input` 字段写这次请求模型的输入，`output` 字段写这次请求我们希望模型输出什么。

> SFT 还有其他数据格式，比如 `ShareGPT`, `ChatML` 等，参考 [datasets-guide](https://docs.unsloth.ai/basics/datasets-guide)

### 4. 本文的任务

本文只有一个微调任务，但是通过三种框架实现。

**一个任务：**

使用 **医疗问答数据集 medical** 对 **Qwen 模型** 做 **SFT 监督微调**。相关资源列表：

- medical 数据集：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)
- LLaMA Factory：
- `Qwen2.5-0.5B-Instruct` 模型：[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- `Qwen2.5-7B-Instruct` 模型：[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

**三种框架：**

- **LLaMA Factory**: 提供简洁的 UI 界面，支持零代码微调大模型
- **trl**: 有 HuggingFace 生态支持，且工具链完备。不仅支持监督微调，对强化学习微调的支持也很好
- **unsloth**: 擅长加速训练和量化技术，能显著减少显存使用量、加快训练速度

本文旨在跑通流程，因此使用 `0.5B` 模型。这既能减少显存占用，也能更快完成任务。如果你有 RTX 5090 或者 GPU 服务器，可以考虑使用 `3B`, `7B` 等更大规模的模型。如果你计划用较多的样本进行训练，可以考虑使用非 Instruct 模型，关于这点建议参考文档 [*Instruct or Base Model?*](https://docs.unsloth.ai/get-started/beginner-start-here/what-model-should-i-use#instruct-or-base-model)

## 二、LLaMA-Factory

首先，我们来安装 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。LLaMA Factory 是一个比较容易上手的微调框架，可以通过 WebUI 来零代码微调大模型。


1. 环境准备
   - 安装 LLaMA Factory
   - 下载 Qwen 模型
   - 模型推理测试
2. 数据准备
   - 下载医疗对话数据集
   - 检查数据格式
   - 添加描述文件
3. 微调大模型
   - SFT 监督微调
   - 加载训练好的 LoRA 文件
   - 导出微调后的模型
4. vLLM 作为推理后端
   - 启动后端推理服务
   - 运行客户端获取结果


## 三、trl

[trl](https://github.com/huggingface/trl) 的功能强大，支持 SFT, PPO, DPO, GRPO 等微调方法。并且有良好的生态支持，比如，trl 可以配合 [peft](https://github.com/huggingface/peft) 的 `LoraConfig` 模块定义 LoRA 参数；配合 [unsloth](https://github.com/unslothai/unsloth) 的 `FastLanguageModel` 模型加载模型。

与上一节的 LLaMA Factory 相比，trl 可以更精细地定义训练中的行为。比如，如何加载数据集、如何构建损失函数、允许哪些参数层参与训练等等。适合需要深度控制训练过程的场景。


1. 加载数据集
2. 微调 Qwen 模型
3. 加载微调后的模型


## 四、unsloth

[unsloth](https://github.com/unslothai/unsloth) 是目前最适合在消费级显卡上使用的微调框架，它的显存消耗少且速度极快。最感人的是它的文档也是最全的，回答了初学者的常见疑惑和一些很有价值的问题 --> [⭐ Beginner? Start here!](https://docs.unsloth.ai/get-started/beginner-start-here)


1. 加载模型
2. 加载数据
3. 微调模型
4. 模型推理
5. 保存模型
6. 保存模型后重新加载
