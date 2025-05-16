import os
import sys

from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer

from infllm.utils import GreedySearch
from infllm.inf_utils import inf_pred, get_model_and_tokenizer
from ourmodel.utils2 import prediction, ori_pred
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
from QA_mlp_trainer import MLPModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--model_center", action="store_true", default=False)
    parser.add_argument('--model_name', type=str, choices=["llama-3-inst","mistral-inst","vicuna","qwen"])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--method', type=str, choices=["our","origin","inf-llm","stream","lm-infinite"])
    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(extra_args)
    conf = OmegaConf.merge(conf, cli_conf)
    conf.model_path = args.model_path
    conf.method = args.method
    conf.datasets = args.datasets
    conf.model_name = args.model_name
    if args.method in ["inf-llm","stream","lm-infinite"]:
        conf.model.model_center = args.model_center

    datasets_str = args.datasets.strip().strip(",")
    datasets_list = datasets_str.split(",")
    conf.datasets = []
    for d in datasets_list:
        conf.datasets.append(d.strip())
    return conf

@torch.no_grad()
def get_pred(tokenizer,model, device, data, max_gen,
             prompt_format, out_path,
             model_name, method,
             tar_len=None, chunk_model=None,MLPmodel=None,bpp_threshold=None,chunk_len=None,
             gen_chunk_size=None,select_method="MLP",overlap=None):

    if method in ["inf-llm","stream","lm-infinite"]:
        searcher = GreedySearch(model, tokenizer)
    for json_obj in tqdm(data):



        if method == "our":
            pred = prediction(model,tokenizer,prompt_format,chunk_model,
                              chunk_len,bpp_threshold,json_obj,model_name,
                              device,max_gen,tar_len,MLPmodel, "longbench",select_method, overlap)
        elif method in ["inf-llm","stream","lm-infinite"]:
            pred = inf_pred(model_name,tokenizer,prompt_format,json_obj,
                            "longbench",searcher,max_gen,gen_chunk_size)
            searcher.clear()
        elif method == "origin":
            pred =ori_pred(model,tokenizer,prompt_format,json_obj,
                           tar_len,device,max_gen,"longbench",model_name)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],
                       "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model_path = args.model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model_name + "-" + args.method
    datasets = args.datasets

    if args.method == "our":
        dataset2prompt = json.load(open("longbench/config/dataset2prompt_2.json", "r"))
        MLPmodel = torch.load(args.MLP_model).to(device)
        MLPmodel.eval()
        if args.use_dynamic:
            chunk_model = SentenceTransformer("minilm/").to(device)
        else:
            chunk_model = None
    else:
        dataset2prompt = json.load(open("longbench/config/dataset2prompt.json", "r"))

    dataset2maxlen = json.load(open("longbench/config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")

    if args.method in ["inf-llm","stream","lm-infinite"]:
        model,tokenizer = get_model_and_tokenizer(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        model = model.eval()


    for dataset in datasets:
        print("working on " + dataset)
        data = load_dataset("json", data_files="longbench/data/" + dataset + ".jsonl", split='train')

        if not os.path.exists(f"longbenchpred/{model_name}"):
            os.makedirs(f"longbenchpred/{model_name}")
        out_path = f"longbenchpred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        overlap =  args.overlap if hasattr(args, 'overlap') else None

        if args.method == "our":
            get_pred(tokenizer, model, device, data, max_gen,
                     prompt_format,out_path, args.model_name,
                     args.method,args.target_length, chunk_model,
                     MLPmodel, args.bpp_threshold, args.chunk_length, select_method=args.select_method,overlap=overlap)
        elif args.method == "origin":
            get_pred(tokenizer, model, device, data, max_gen,
                     prompt_format, out_path, args.model_name,
                     args.method,args.target_length)
        elif args.method in ["inf-llm","stream","lm-infinite"]:
            get_pred(tokenizer, model, device, data, max_gen,
                     prompt_format, out_path, args.model_name,
                     args.method,gen_chunk_size=args.chunk_size)
