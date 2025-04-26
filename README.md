# LLaMA-Factory 微调大模型

> 大模型的知识点太多，一不小心就学杂了：推理框架 vLLM、量化方法 AWQ、微调技术 QLoRA、网络搜索 SearXNG、工具调用 MCP Server、检索增强生成 GraphRAG、向量数据库 Chroma ... 跟报菜名似的。这就是大模型的特点，它是一个复杂工程。

基座大模型，比如 Qwen, Llama，有很强的通用能力。但在特定领域，它的表现可能不如领域小模型。为了提高大模型在特定领域的表现，可以用领域数据集对大模型进行微调，以增强模型对 领域知识、语言风格 和 输出格式 的记忆。

目前，微调大致可分三种技术路线：

1. **无监督 / 自监督微调**：通过降低预训练目标的 loss 继续训练
2. **监督微调**：使用显式标签数据（如问答数据集）优化模型
3. **强化学习微调**：通过 reward 引导模型进行优化
    - RLHF
    - PPO
    - DPO
    - GRPO

这里比较简单的是 **监督微调 (Supervised Fine-Tuning, SFT)**。作为入门学习项目，当然选简单的啦。接下来，我们要用医疗数据集 **medical** 通过 **LLaMA Factory** 对 Qwen 模型进行 **SFT 微调**。

- medical 数据集：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)
- Qwen 模型：[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

## 一、LLaMA Factory

> 微调相关的框架是相当多，比如 [unsloth](https://github.com/unslothai/unsloth), [trl](https://github.com/huggingface/trl), [peft](https://github.com/huggingface/peft)。跟前面这些比起来，LLaMA Factory 是最容易上手的。

[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 支持通过 WebUI 零代码微调大语言模型。它的功能还包括加速算子、量化、实验监控、效果评估等，甚至带了一个基于 gradio 开发的 ChatBot 推理页面。

1. 安装 LLaMA Factory
    - 安装 CUDA
    - 安装 LLaMA-Factory
    - 安装 bitsandbytes
2. 下载 Qwen 模型
    - 安装 ModelScope
    - 下载 Qwen2.5-7B-Instruct 模型
3. 模型推理测试
    - 使用 transformers 库推理
    - 使用 ChatBot 推理

## 二、数据准备

1. 下载数据集
2. 数据集格式
3. 填写描述文件

## 三、微调大模型







参考：

- [入门教程](https://zhuanlan.zhihu.com/p/695287607)
- [框架文档](https://llamafactory.readthedocs.io/zh-cn/latest/)
