python QA_data_process.py \
--data_path adversarialqa/train.json \
--data_output QA_train_data_llama3.pt \
--label_output QA_train_label_llama3.pt \
--model_path your LLM model path

python QA_data_process.py \
--data_path adversarialqa/test.json \
--data_output QA_train_test_llama3.pt \
--label_output QA_train_test_llama3.pt \
--model_path your LLM model path

python QA_data_process.py \
--data_path adversarialqa/dev.json \
--data_output QA_val_data_llama3.pt \
--label_output QA_val_label_llama3.pt \
--model_path your LLM model path
