import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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

    # 加载模型，并使用 float16 进行推理，device_map="auto" 让 transformers 自动选择设备（CPU/GPU）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,  # 传入修改后的配置
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Model loaded successfully!")
    return model, tokenizer


def chat_with_model(model, tokenizer):
    """
    交互式聊天功能，使用流式输出。
    """
    print("\nWelcome to DeepSeek-R1 Chat! Type 'exit' to quit.")

    history = []  # 记录对话历史

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Exiting chat...")
            break

        # 处理输入文本
        history.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 使用流式推理生成输出
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=16384,  # 限制最大生成长度
                do_sample=True,  # 允许采样（更具创造性）
                temperature=0.7,  # 控制随机性，值越低输出越确定
                top_k=50,  # 限制候选词汇范围
                top_p=0.9,  # 仅考虑概率总和达到 p 的候选词
                pad_token_id=tokenizer.eos_token_id  # 避免生成无意义的填充内容
            )

        # 解码输出（流式打印）
        response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

        print("AI:", end="", flush=True)
        for char in response:
            print(char, end="", flush=True)
            torch.cuda.synchronize() if torch.cuda.is_available() else None  # 保证流式输出不卡顿
        print()

        # 记录对话历史
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    model, tokenizer = load_model()
    chat_with_model(model, tokenizer)
