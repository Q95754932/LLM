import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import TextIteratorStreamer

# 指定模型名称（Hugging Face 上的路径）
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def load_model():
    """
    加载 DeepSeek-R1-Distill-Qwen-1.5B 模型和分词器。
    """
    print("Loading model... This may take a few minutes.")

    # 加载 tokenizer（用于将文本转换为模型输入）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 获取模型配置，并手动禁用 Sliding Window Attention
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.sliding_window = None  # 显式禁用 Sliding Window Attention

    # 加载模型，并使用 float32 进行推理，device_map="auto" 让 transformers 自动选择设备（CPU/GPU）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,  # 传入修改后的配置
        torch_dtype=torch.float32,
        device_map="auto",  # 需要安装 accelerate 库
    )

    print("Model loaded successfully!")
    return model, tokenizer


def chat_with_model(model, tokenizer):
    """
    使用 TextIteratorStreamer 实现交互式聊天功能，真正的流式输出。
    """
    print("\nWelcome to DeepSeek-R1 Chat! Type 'exit' to quit.")

    history = []  # 记录对话历史

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Exiting chat...")
            break

        # 将用户输入加入对话历史
        history.append({"role": "user", "content": user_input})

        # 这里假设您有自己的对话模板方法 apply_chat_template
        # 如果没有，可以直接使用 user_input 构造 prompt
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 创建一个 TextIteratorStreamer 来接收流式输出
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True  # 跳过直接打印 Prompt  # 跳过特殊token
        )

        # 配置生成参数
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=16384,  # 限制最大生成长度，可根据需求调整
            do_sample=True,  # 允许采样（更具创造性）
            temperature=0.7,  # 控制随机性，值越低输出越确定
            top_k=50,  # 限制候选词汇范围
            top_p=0.9,  # 仅考虑概率总和达到 p 的候选词
            pad_token_id=tokenizer.eos_token_id,  # 避免生成无意义的填充内容
            streamer=streamer  # 指定流式输出
        )

        # 启动一个线程来异步生成文本
        generation_thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        generation_thread.start()

        print("AI: ", end="", flush=True)

        # 从 streamer 中读取模型流式输出的文本
        partial_response = []
        for new_text in streamer:
            print(new_text, end="", flush=True)
            partial_response.append(new_text)

        # 等待生成线程结束
        generation_thread.join()
        print()

        # 将本轮生成的完整文本保存到 history
        response_text = "".join(partial_response)
        history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    model, tokenizer = load_model()
    chat_with_model(model, tokenizer)
