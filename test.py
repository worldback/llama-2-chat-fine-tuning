from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

def get_model_and_tokenizer(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return model, tokenizer, device

def process_input(conv: list):
    chat_tokens = tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt")
    return chat_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Llama-2 model with optional fine-tuned model.")
    parser.add_argument("--path", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="The model ID or path to the fine-tuned model. Default is 'meta-llama/Llama-2-7b-hf'.")
    args = parser.parse_args()
    model_id = args.path
    model, tokenizer, device = get_model_and_tokenizer(model_id)
    system_msg = """
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"""
    conv = [{"role": "system", "content": system_msg}]
    with torch.inference_mode():
        while True:
            msg =input("user:").strip()
            if msg == "q":
                break
            conv.append({"role": "user", "content": msg})
            input_ids = process_input(conv)
            if len(input_ids[0]) > 1024:
                # 微调时使用了 1024的最大字符，推理时或许可以超过，请自行测试效果。
                print("out of context window 1024, run over.")
                break
            output = model.generate(input_ids=input_ids.to(device), max_new_tokens=300, num_return_sequences=1)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            prompt_length = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
            print(f"assistant:{generated_text[prompt_length:]}")
            conv.append({"role": "assistant", "content": generated_text[prompt_length:]})