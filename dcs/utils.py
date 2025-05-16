import torch
import torch.nn.functional as F
from torch import nn
from QA_mlp_trainer import MLPModel

from lveval.utils import model_generate, post_process
from sentence_chunk import sentence_chunking_main


def build_chat(tokenizer, prompt, basic_model_name, data_name=None):
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
        prompt = prompt


    return prompt

def cos_sim_cal(mat1,mat2):
    mat2 = torch.squeeze(mat2, dim=0)
    mat = torch.mul(mat1, mat2)
    mat = torch.sum(mat,dim=-1)
    mat1_norm = torch.norm(mat1, dim=-1)
    mat2_norm = torch.norm(mat2, dim=-1)
    mat_norm = torch.mul(mat1_norm, mat2_norm)
    return torch.mean(mat / mat_norm)

def get_chunk(json_obj,chunk_size,tokenizer=None):
    chunks = list()
    try:
        txt = json_obj['context']
    except:
        txt = json_obj

    tokenized_txt = tokenizer(txt, truncation=False, return_tensors="pt").input_ids[0]
    while True:
        chunk = tokenizer.decode(tokenized_txt[:min(chunk_size, len(txt))], skip_special_tokens=True)
        chunks.append(chunk)
        if tokenized_txt.size(-1) >= chunk_size:
            tokenized_txt = tokenized_txt[chunk_size:]
        else:
            break
    return chunks


def llm_chunk_process(input_chunks,bl):
    chunk_count = len(input_chunks)
    arr = list(range(chunk_count))
    arr = arr[:int(chunk_count // bl//2)]+arr[-int(chunk_count // bl//2):]

    return arr


@torch.no_grad()
def chunk_selection(input_chunks, model, mlpmodel, instruct,scaling,select_method="MLP"):
    leng = instruct.input_ids.size(-1)
    for chunk in input_chunks:
        input_ids = torch.cat((chunk.input_ids, instruct.input_ids), dim=-1)
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
        )
        attn = outputs.attentions[-1]
        attn = torch.mean(attn, dim=1)
        attn = attn[:, -leng:, :-leng]
        attn = torch.mean(attn, dim=1).squeeze(0)
        attn = F.softmax(attn, dim=0)
        outputs = model(
            input_ids=chunk.input_ids,
            output_hidden_states=True,
        )

        hidden_states = torch.cat((outputs.hidden_states[-1][:, :1, :],
                                   torch.mm(attn.unsqueeze(0), outputs.hidden_states[-1].squeeze(0)).unsqueeze(0),
                                   outputs.hidden_states[-1][:, -1:, :]), dim=1)
        try:
            chunks_avg_hidden_states = torch.cat((chunks_avg_hidden_states,hidden_states),dim=0)
        except:
            chunks_avg_hidden_states = hidden_states

    outputs = model(
        **instruct,
        output_hidden_states=True,
        output_attentions=True,
    )
    attn = outputs.attentions[-1]
    attn = torch.mean(attn, dim=1)
    attn = torch.mean(attn, dim=1).squeeze(0)
    attn = F.softmax(attn, dim=0)

    ins_hidden_states = torch.cat((outputs.hidden_states[-1][:, :1, :],
                                   torch.mm(attn.unsqueeze(0), outputs.hidden_states[-1].squeeze(0)).unsqueeze(0),
                                   outputs.hidden_states[-1][:, -1:, :]), dim=1)

    similarity = list()
    if select_method == "MLP":
        for hidden_states in chunks_avg_hidden_states:
            input_tensor = torch.cat((hidden_states.unsqueeze(0), ins_hidden_states), dim=1)
            output = mlpmodel(input_tensor)
            output = F.softmax(output[0])
            similarity.append(output[1])

    else:
        for hidden_states in chunks_avg_hidden_states:
            cos = cos_sim_cal(hidden_states,ins_hidden_states)
            similarity.append(cos.tolist())

    similarity = torch.tensor(similarity)

    sorted_similarity = torch.sort(similarity, descending=True).indices.squeeze(0)

    try:
        selected_chunks = sorted_similarity[:int(len(sorted_similarity) // scaling)]
    except:
        selected_chunks = sorted_similarity.unsqueeze(0)


    selected_chunks = torch.sort(selected_chunks).values.tolist()
    return selected_chunks,torch.sort(similarity, descending=True).values.tolist()



def chunk_splicing(init, chunks, selected_chunks, instruct):
    text = init
    for chunk_id in selected_chunks:
        text += chunks[chunk_id]
        text += " "
    text += instruct

    return text

@torch.no_grad()
def generate(model,tokenizer,max_gen, tokenized_prompt,context_length):
    output = model.generate(
        **tokenized_prompt,
        max_new_tokens=max_gen,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )[0]
    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

    return pred

def prediction(model,tokenizer, prompt_format,chunk_model,
                     chunk_len,bpp_threshold,json_obj,model_name,
                     device,max_gen, tar_len, MLPmodel, data_name, select_method="MLP"):

    prompt = prompt_format.format(**json_obj)

    if chunk_model:
        chunks, _ = sentence_chunking_main(chunk_len, bpp_threshold, True, json_obj['context'], chunk_model)
    else:
        chunks = get_chunk(json_obj, chunk_len,tokenizer)

    prompt = prompt.split("####")

    total_len = 0
    input_chunks = list()
    for chunk in chunks:
        input = tokenizer(chunk, truncation=False, return_tensors="pt").to(device)
        input_chunks.append(input)
        total_len += len(input.input_ids[0])
    scaling = max(total_len / tar_len, 1)
    instruct = tokenizer(prompt[1], truncation=False, return_tensors="pt").to(device)
    if select_method == "MLP":
        selected_chunks, _ = chunk_selection(input_chunks, model, MLPmodel, instruct, scaling)
    elif select_method == "cos":
        selected_chunks, _ = chunk_selection(input_chunks, model, MLPmodel, instruct, scaling, "cos")
    selected_prompt = chunk_splicing(prompt[0], chunks, selected_chunks, prompt[1])
    selected_prompt = build_chat(tokenizer, selected_prompt, model_name, data_name)

    if data_name == "longbench":

        if model_name == 'mistral-inst':
            add_special_tokens = False
        else:
            add_special_tokens = True

        tokenized_prompt = tokenizer(selected_prompt, truncation=False, return_tensors="pt",
                                     add_special_tokens=add_special_tokens).to(device)
        context_length = tokenized_prompt.input_ids.shape[-1]

        pred = generate(model,tokenizer,max_gen, tokenized_prompt,context_length)
        torch.cuda.empty_cache()
        return pred

    elif data_name == "lveval":
        pred = model_generate(tokenizer, selected_prompt, max_gen, model,model_name)
        pred = post_process(pred, model_name)
        torch.cuda.empty_cache()
        return pred


def ori_pred(model,tokenizer,prompt_format,json_obj,tar_len,
             device,max_gen,data_name,model_name):
    prompt = prompt_format.format(**json_obj)

    if data_name == "longbench":
        if model_name == 'mistral-inst':
            add_special_tokens = False
        else:
            add_special_tokens = True

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt",
                                    add_special_tokens=add_special_tokens).input_ids[0]

        if len(tokenized_prompt) > tar_len:
            half = int(tar_len / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True)

        prompt = build_chat(tokenizer, prompt, model_name, data_name)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt",
                                     add_special_tokens=add_special_tokens).to(device)

        context_length = tokenized_prompt.input_ids.shape[-1]
        pred = generate(model, tokenizer, max_gen, tokenized_prompt, context_length)

        return pred
