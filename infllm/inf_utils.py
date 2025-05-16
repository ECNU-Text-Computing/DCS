import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from infllm import patch_model_center, patch_hf


def get_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.path)
    if config.model_center:
        import bmtrain as bmt
        bmt.init_distributed(seed=233)
        from model_center.model import Llama, LlamaConfig
        model_config = LlamaConfig.from_pretrained(config.path)
        model_config.dtype = torch.bfloat16
        model = Llama(model_config)
        bmt.load(model, os.path.join(config.path, "pytorch_model.pt"), strict=False)
        model = patch_model_center(model, config.type, **config)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.path,
                                                     torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                     device_map="cuda")
        model = patch_hf(model, config.type, **config)
    return model, tokenizer

def build_chat(tokenizer, prompt, basic_model_name, data_name):
    model_name = basic_model_name.strip().lower()
    if model_name == "llama-3-inst" and data_name == "longbench":
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif model_name == "qwen":
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif model_name == "mistral-inst":
        prompt = f"<s>[INST] {prompt} [/INST]"
    elif model_name == "vicuna":
        from fastchat.conversation import get_conv_template
        conv = get_conv_template("vicuna_v1.1")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    else:
        prompt=prompt

    return prompt

def inf_pred(model_name,tokenizer,prompt_format,json_obj,data_name,searcher,max_gen,gen_chunk_size):
    prompt = prompt_format.format(**json_obj)
    extra_end_token_ids = []
    if model_name == "llama-3-inst":
        extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])
    if model_name == "qwen":
        extra_end_token_ids.append(tokenizer.encode("<|im_end|>", add_special_tokens=False)[0])

    prompt = build_chat(tokenizer, prompt, model_name, data_name)

    if data_name == "longbench":
        if model_name.strip().lower() in ['mistral-inst']:
            add_special_tokens = False
        else:
            add_special_tokens = True

    else:
        add_special_tokens = True

    tokenized_prompt = \
        tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]

    output = searcher.generate(
        input_ids=tokenized_prompt,
        max_length=max_gen,
        chunk_size=gen_chunk_size,
        extra_end_token_ids=extra_end_token_ids
    )
    return output[0]