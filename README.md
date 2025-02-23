# 🚀 超简单本地部署 DeepSeek-R1！

## 📌 项目简介

本项目演示了如何 **使用不到 100 行代码，在本地轻松部署 DeepSeek-R1-Distill-Qwen-1.5B 模型**，实现高效的 AI 交互。

💡 **核心特点：**
- **极简部署**：只需不到 100 行代码，即可在本地运行强大的 AI 大模型。
- **零配置上手**：无需繁琐环境配置，安装依赖后直接运行。
- **流式输出**：基于 `TextIteratorStreamer` 实现实时流式输出，像 ChatGPT 一样边打字边显示。
- **自动选择设备**：代码会自动检测 GPU / CPU，确保最优推理性能。
- **低门槛训练**：无需复杂配置，新手也能轻松上手！

🎯 **你还在等什么？快来试试吧！**

---

## 🔥 一分钟极速部署

### 1️⃣ 安装环境（只需 1 行命令）
在运行本项目之前，确保你的环境已经安装了以下依赖项：

```bash
pip install torch transformers accelerate
```

💡 **加速建议**：
- **NVIDIA GPU 用户**：建议安装 **CUDA 版本的 PyTorch** 以获得更快推理速度：[PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)
- **Mac M1/M2 用户**：使用 Metal 版本加速：
  
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

### 2️⃣ 运行代码（直接执行，无需改动）
下载或复制 `qwen1.5B_stream.py` 代码，并运行：
```bash
python qwen1.5B_stream.py
```
等待模型加载完成后，你就可以开始和 AI 进行交互！

---

## 📜 代码解析：简单到离谱！

### 🎯 主要功能
本项目的核心功能包括：
✅ **零门槛部署，安装即用**
✅ **流式输出 AI 回复，媲美在线 AI 体验**
✅ **无需手动设置参数，默认优化推理**
✅ **自动选择 GPU / CPU，性能最大化**