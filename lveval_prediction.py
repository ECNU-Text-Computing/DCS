import os
import re
import torch
import argparse

from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from QA_mlp_trainer import MLPModel
from infllm import GreedySearch
from infllm.inf_utils import get_model_and_tokenizer, inf_pred

from lveval.config import (
    DATASET_MAXGEN, 
    MY_DATASET_PROMPT,
    DATASET_PROMPT,
)
from lveval.utils import (
    seed_everything,
    get_dataset_names,
    dump_preds_results,
    load_LVEval_dataset,
    truncate_prompt,
    build_chat,
    model_generate,
    post_process,
)

from sentence_transformers import SentenceTransformer
from dcs.utils import prediction


def get_pred(model,tokenizer,data,max_gen,prompt_format,model_name,method,
             tar_len=None, chunk_model=None,MLPmodel=None,bpp_threshold=None,
             chunk_len=None,gen_chunk_size=None,select_method="MLP"):
    preds = []

    if method in ["inf-llm","stream","lm-infinite"]:
        searcher = GreedySearch(model, tokenizer)

    for json_obj in tqdm(data):
        if method == "our":
            pred = prediction(model,tokenizer, prompt_format,chunk_model,
                         chunk_len,bpp_threshold,json_obj,model_name,
                         device,max_gen, tar_len, MLPmodel, "lveval",select_method)
        elif method in ["inf-llm","stream","lm-infinite"]:
            pred = inf_pred(model_name, tokenizer, prompt_format, json_obj,
                            "lveval", searcher, max_gen, gen_chunk_size)
            searcher.clear()
        elif method == "origin":
            prompt = prompt_format.format(**json_obj)
            prompt = truncate_prompt(tokenizer, prompt, tar_len)
            prompt = build_chat(tokenizer, prompt, model_name)
            pred = model_generate(tokenizer, prompt, max_gen, model,model_name)
            pred = post_process(pred, model_name)

        preds.append({
            "pred": pred,
            "answers": json_obj["answers"],
            "gold_ans": json_obj["answer_keywords"] if "answer_keywords" in json_obj else None,
            "input": json_obj["input"],
            "all_classes": json_obj["all_classes"] if "all_classes" in json_obj else None,
            "length": json_obj["length"],
        })
    return preds

def single_processing(model, tokenizer, datasets, out_path, args,chunk_model=None,MLPmodel=None):
    for dataset in tqdm(datasets):
        datas = load_LVEval_dataset(dataset, args.data_path)
        dataset_name = re.split('_.{1,3}k', dataset)[0]
        if args.method == "our":
            prompt_format = MY_DATASET_PROMPT[dataset_name]
        else:
            prompt_format = DATASET_PROMPT[dataset_name]

        max_gen = DATASET_MAXGEN[dataset_name]
        if args.method == "our":
            preds = get_pred(model,tokenizer,datas,max_gen,prompt_format,
                             args.model_name,args.method,
                             tar_len=args.target_length,chunk_model=chunk_model,MLPmodel=MLPmodel,
                             chunk_len =args.chunk_length,bpp_threshold =args.bpp_threshold,select_method=args.select_method)
        elif args.method in ["inf-llm","stream","lm-infinite"]:
            preds = get_pred(model, tokenizer, datas, max_gen, prompt_format,
                             args.model_name, args.method,gen_chunk_size=args.chunk_size)
        elif args.method == "origin":
            preds = get_pred(model, tokenizer, datas, max_gen, prompt_format,
                             args.model_name, args.method,tar_len=args.target_length)

        dump_preds_results(preds, os.path.join(out_path, dataset + ".jsonl"))

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model_path', type=str, default="/home/bhsheng/LLMmodel/Llama-2-7b-chat-hf")
    # parser.add_argument('--model_path', type=str, default="/home/bhsheng/LLMmodel/Mistral-7B-Instruct-v0.1")
    parser.add_argument('--model_path', type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--model_center", action="store_true", default=False)
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--data_length', type=str, default=None)
    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(extra_args)
    conf = OmegaConf.merge(conf, cli_conf)
    conf.model_path = args.model_path
    conf.method = args.method
    conf.datasets = args.datasets
    conf.data_path = args.data_path
    conf.model_name = args.model_name
    if args.method in ["inf-llm", "stream", "lm-infinite"]:
        conf.model.model_center = args.model_center

    conf.data_length = args.data_length.split(",")
    datasets_str = args.datasets.strip().strip(",")
    datasets_list = datasets_str.split(",")
    conf.datasets = []
    for d in datasets_list:
        conf.datasets.append(d.strip())
    return conf

if __name__ == "__main__":
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    model_path = args.model_path

    datasets = get_dataset_names(args.datasets, args.data_length)
    model_name = args.model_name + "-" + args.method

    if not os.path.exists("lveval_pred"):
        os.makedirs("lveval_pred")


    if args.method == "our":
        MLPmodel = torch.load(args.MLP_model).to(device)
        MLPmodel.eval()
        if args.use_dynamic:
            chunk_model = SentenceTransformer("minilm/").to(device)
        else:
            chunk_model = None

    if not os.path.exists(f"lveval_pred/{model_name}"):
        os.makedirs(f"lveval_pred/{model_name}")

    out_path = f"lveval_pred/{model_name}"

    if args.method in ["inf-llm","stream","lm-infinite"]:
        model,tokenizer = get_model_and_tokenizer(args.model)
        single_processing(model, tokenizer, datasets, out_path, args)
    elif args.method == "our":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        model = model.eval()
        single_processing(model, tokenizer, datasets, out_path, args, chunk_model, MLPmodel)
    elif args.method == "origin":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        model = model.eval()
        single_processing(model, tokenizer, datasets, out_path, args)


