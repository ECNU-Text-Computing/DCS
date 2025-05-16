# DCS
Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models

## How to run our codes

### 1.Install the requirements
python=3.10

Install PyTorch according to your CUDA version.
```sh
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

Install the requirements with pip: 
```sh
pip install -r requirements.txt
```




### 2. Download datasets and models form the following sites.
adversarialqa: https://adversarialqa.github.io/

coqa: https://stanfordnlp.github.io/coqa/

LongBench: https://huggingface.co/datasets/THUDM/LongBench

LVEval: https://huggingface.co/datasets/Infinigence/LVEval

MiniLM: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

SQuAD: https://rajpurkar.github.io/SQuAD-explorer/

### 3. Run QA_data_process.py

Run QA_data_process.py to generate MLP training data. 

```sh
bash qadata_l3.sh
```

### 4. Evaluate DCS on LongBench

```sh
bash longbench_l3_our.sh
```

### 5. Evaluate DCS on LVEval

```sh
bash lveval_l3_our.sh
```
