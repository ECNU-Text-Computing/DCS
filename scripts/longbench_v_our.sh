export CUDA_VISIBLE_DEVICES="4"
config=dcs/config/vicuna-our.yaml

#datasets="multifieldqa_en,narrativeqa,qasper,hotpotqa,2wikimqa,musique,"

datasets="multifieldqa_en"

python QA_mlp_trainer.py \
--dim1 4096 \
--dim2 1024 \
--epoch 20 \
--model_name vicuna \
--lr 0.000015 \
--model_path clfmodel/clfmodel_v_mlp.pth

python longbench_prediction.py \
--config_path ${config} \
--datasets ${datasets} \
--model_path your LLM model path \
--method our \
--model_name vicuna

python longbench_eval.py \
--model vicuna-our