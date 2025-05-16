import argparse
import json
import random
import sys

import os

import numpy as np
#os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch.nn.functional as F
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="drop/drop_dataset_dev.json")
    parser.add_argument('--data_output', type=str, default="QA_test_data_mistral_d.pt")
    parser.add_argument('--label_output', type=str, default="QA_test_label_mistral_d.pt")
    parser.add_argument('--model_path', type=str, default=None)#Qwen-7B-Chat

    return parser.parse_args(args)


def data_saving(data, file_name):
    torch.save(data, "MLP_train_data/" + file_name)
    print(file_name + " has been saved")

class DataReader:
    def __init__(self,data_path,neg_num):
        self.data_path = data_path
        self.neg_num = neg_num
        self.data = self.dataloading()
        self.qas_list,self.passage_list = self.dataprocessing()
        self.negative_sampling()

    def dataloading(self):
        with open(self.data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            try:
                data = data["data"]
            except:
                data = data
        return data

    def dataprocessing(self):
        prompt ="\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {}\n\nAnswer:"
        qas_list = list()
        passage_list = list()
        passage_count = 0
        data_name = self.data_path.split("/")[0]

        if data_name == "coqa":
            for text in self.data:
                qas = text["questions"]
                passage_list.append(text["story"])
                for qas_isi in qas:
                    # qas_list.append([prompt.format(qas_isi["question"]),qas_isi["is_impossible"],passage_count])
                    if len(qas_isi["input_text"]) >= 30:
                        qas_list.append([prompt.format(qas_isi["input_text"]), passage_count])
                passage_count += 1
        elif data_name == "adversarialqa":
            for text in self.data:
                paragraphs = text["paragraphs"]
                for paragraph in paragraphs:
                    qas = paragraph["qas"]
                    passage_list.append(paragraph["context"])
                    for qas_isi in qas:
                        qas_list.append([prompt.format(qas_isi["question"]),passage_count])
                    passage_count += 1
        elif data_name == "squad":
            for text in self.data:
                paragraphs = text["paragraphs"]
                for paragraph in paragraphs:
                    qas = paragraph["qas"]
                    passage_list.append(paragraph["context"])
                    i=0
                    for qas_isi in qas:
                        if i > 1:
                            break
                        if qas_isi["is_impossible"]==False:
                            qas_list.append([prompt.format(qas_isi["question"]),passage_count])
                            i+=1
                    passage_count += 1
        elif data_name == "drop":
            for text in self.data.values():
                paragraph = text["passage"]
                passage_list.append(paragraph)
                qas = text["qa_pairs"]
                i = 0
                for qas_isi in qas:
                    if i > 9:
                        break
                    qas_list.append([prompt.format(qas_isi["question"]),passage_count])
                    i += 1
                passage_count += 1
        return qas_list, passage_list

    def negative_sampling(self):
        max_num = len(self.passage_list)-1
        pos_qas_list = list()
        neg_qas_list = list()
        for qas in self.qas_list:
            pos_qas_list.append([qas[0], 1, qas[1]])
            for i in range(self.neg_num):
                neg_pass = qas[1]
                while neg_pass == qas[1]:
                    neg_pass = random.randint(0,max_num)
                neg_qas_list.append([qas[0],0,neg_pass])

        self.qas_list = pos_qas_list + neg_qas_list


class LLMEncoder:
    def __init__(self,model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model,self.tokenizer = self.model_loading()

    def model_loading(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path,trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).to(self.device)
        model = model.eval()
        return model, tokenizer

    @torch.no_grad()
    def context_encoding(self, context, qas):
        tokenized_context = self.tokenizer(context, truncation=False, return_tensors="pt").to(self.device)
        tokenized_qas = self.tokenizer(qas, truncation=False, return_tensors="pt").to(self.device)
        leng = tokenized_qas.input_ids.size(-1)
        input_ids = torch.cat((tokenized_context.input_ids, tokenized_qas.input_ids), dim=-1)
        outputs = self.model(
            input_ids=input_ids,
            output_attentions=True,
        )
        attn = outputs.attentions[-1]
        attn = torch.mean(attn, dim=1)
        attn = attn[:, -leng:, :-leng]
        attn = torch.mean(attn, dim=1).squeeze(0)
        attn = F.softmax(attn, dim=0)

        outputs = self.model(
            input_ids=tokenized_context.input_ids,
            output_hidden_states=True,
        )
        last_hidden_states = outputs.hidden_states[-1]
        avg_hidden_states = torch.mm(attn.unsqueeze(0), last_hidden_states.squeeze(0)).unsqueeze(0)
        first_hs = last_hidden_states[:,:1,:]
        avg_hs = avg_hidden_states
        last_hs = last_hidden_states[:,-1:,:]
        hidden_states = torch.cat((first_hs,avg_hs,last_hs),dim=1)

        return hidden_states

    @torch.no_grad()
    def qas_encoding(self,qas):
        tokenized_qas = self.tokenizer(qas, truncation=False, return_tensors="pt").to(self.device)
        outputs = self.model(
            **tokenized_qas,
            output_hidden_states=True,
            output_attentions=True,
        )
        attn = outputs.attentions[-1]
        attn = torch.mean(attn, dim=1)
        attn = torch.mean(attn, dim=1).squeeze(0)
        attn = F.softmax(attn, dim=0)
        last_hidden_states = outputs.hidden_states[-1]
        avg_hidden_states = torch.mm(attn.unsqueeze(0), last_hidden_states.squeeze(0)).unsqueeze(0)
        first_hs = last_hidden_states[:, :1, :]
        avg_hs = avg_hidden_states
        last_hs = last_hidden_states[:, -1:, :]
        hidden_states = torch.cat((first_hs, avg_hs, last_hs), dim=1)

        return hidden_states


    def data_processing(self,qas_list, context_list):
        label_list = list()
        hidden_states_list = None
        for data in tqdm(qas_list):
            qas_hidden_states = self.qas_encoding(data[0])
            con_hidden_states = self.context_encoding(context_list[data[-1]],data[0])
            hidden_states = torch.cat((con_hidden_states,qas_hidden_states),dim=1)
            try:
                hidden_states_list = torch.cat((hidden_states_list,hidden_states),dim=0)
            except:
                hidden_states_list = hidden_states
            if data[1]:
                label_list.append(1)
            else:
                label_list.append(0)

        return hidden_states_list,torch.tensor(label_list)


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists("MLP_train_data"):
        os.makedirs("MLP_train_data")

    sqreader = DataReader(args.data_path,1)
    encoder = LLMEncoder(args.model_path)

    data, label = encoder.data_processing(sqreader.qas_list,sqreader.passage_list)

    data_saving(data,args.data_output)
    data_saving(label,args.label_output)