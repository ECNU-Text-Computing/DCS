export CUDA_VISIBLE_DEVICES="0"
config=dcs/config/llama-3-our.yaml

datasets="hotpotwikiqa_mixup,loogle_SD_mixup,loogle_CR_mixup,loogle_MIR_mixup,multifieldqa_en_mixup,factrecall_en"

python lveval_prediction.py \
--config_path ${config} \
--datasets ${datasets} \
--model_path your LLM path \
--method our \
--model_name llama-3-inst \
--data_path lveval/data/ \
--data_length "16k"

python lveval_evaluation.py \
--input_dir lveval_pred/llama-3-inst-our