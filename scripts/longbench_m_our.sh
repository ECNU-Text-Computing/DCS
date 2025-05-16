export CUDA_VISIBLE_DEVICES="5"
config=dcs/config/mistral-our.yaml

#datasets="multifieldqa_en,narrativeqa,qasper,hotpotqa,2wikimqa,musique,"

datasets="multifieldqa_en"

python QA_mlp_trainer.py \
--dim1 4096 \
--dim2 256 \
--epoch 10 \
--model_name mistral \
--lr 0.00001 \
--model_path clfmodel/clfmodel_m_mlp.pth

python longbench_prediction.py \
--config_path ${config} \
--datasets ${datasets} \
--model_path your LLM model path \
--method our \
--model_name mistral-inst

python longbench_eval.py \
--model mistral-inst-our