model_1_list="Mistral-7B-Instruct-v0.3 Mistral-7B-Instruct-v0.2 Mistral-7B-Instruct-v0.1 Llama-2-7B-Chat Llama-3-8B-Instruct Llama-3.2-3B-Instruct"
model_2_list="Mistral-7B-Instruct-v0.3 Mistral-7B-Instruct-v0.2 Mistral-7B-Instruct-v0.1 Llama-2-7B-Chat Llama-3-8B-Instruct Llama-3.2-3B-Instruct"

for model_1 in $model_1_list; do
    for model_2 in $model_2_list; do
        python affine_transformation.py --model "${model_1}_None_0_${model_2}_None_0" --setting A_forward
        python affine_transformation.py --model "${model_1}_None_0_${model_2}_None_0" --setting B_forward
    done
done
