export CUDA_VISIBLE_DEVICES="2"
config=dcs/config/llama-3-our.yaml

#datasets="multifieldqa_en,narrativeqa,qasper,hotpotqa,2wikimqa,musique,"

datasets="multifieldqa_en"

python QA_mlp_trainer.py \
--dim1 8192 \
--dim2 1024 \
--epoch 20 \
--model_name llama3 \
--lr 0.00001 \
--model_path clfmodel/clfmodel_l3_mlp.pth

python longbench_prediction.py \
--config_path ${config} \
--datasets ${datasets} \
--model_path your LLM model path \
--method our \
--model_name llama-3-inst

python longbench_eval.py \
--model llama-3-inst-our


