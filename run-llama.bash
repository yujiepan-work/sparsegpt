mkdir ./llama-7b-dense
CUDA_VISIBLE_DEVICES=2 python llama.py decapoda-research/llama-7b-hf c4  2>&1 | tee ./llama-7b-dense/llama-7b-unstructured50.dense.log &

mkdir ./llama-7b-unstructured50
CUDA_VISIBLE_DEVICES=0 python llama.py decapoda-research/llama-7b-hf c4 --sparsity .5  2>&1 | tee ./llama-7b-unstructured50/log.log &

mkdir ./llama-7b-8bit-unstructured50
CUDA_VISIBLE_DEVICES=1 python llama.py decapoda-research/llama-7b-hf c4 --sparsity .5 --wbits 8  2>&1 | tee ./llama-7b-8bit-unstructured50/log.log &

mkdir ./llama-7b-4bit-unstructured50
CUDA_VISIBLE_DEVICES=3 python llama.py decapoda-research/llama-7b-hf c4 --sparsity .5 --wbits 4  2>&1 | tee ./llama-7b-4bit-unstructured50/log.log &
