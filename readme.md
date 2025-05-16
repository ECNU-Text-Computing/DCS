# Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models

## How to run our codes

### 1.Install the requirements
python=3.10

Install the requirements with pip: 
```sh
pip install -r requirements.txt
```

### 1. Download datasets and models form the following sites.
adversarialqa: https://adversarialqa.github.io/

coqa: https://stanfordnlp.github.io/coqa/

LongBench: https://huggingface.co/datasets/THUDM/LongBench

LVEval: https://huggingface.co/datasets/Infinigence/LVEval

MiniLM: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

SQuAD: https://rajpurkar.github.io/SQuAD-explorer/

### 2. Run QA_data_process.py

Run QA_data_process.py to generate MLP training data. 

```sh
bash qadata_l3.sh
```

### 3. Evaluate DCS on LongBench

```sh
bash longbench_l3_our.sh
```

### 3. Evaluate DCS on LVEval

```sh
bash lveval_l3_our.sh
```
