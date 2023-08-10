# mkdir ./opt-13b-unstructured50-cali-by-c4
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-13b c4 --sparsity .5 --save ./opt-13b-unstructured50-cali-by-c4 2>&1 | tee ./opt-13b-unstructured50-cali-by-c4/log.log &

# mkdir ./opt-13b-8bit-unstructured50-cali-by-c4
# CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-13b c4 --sparsity .5 --wbits 8 --save ./opt-13b-8bit-unstructured50-cali-by-c4 2>&1 | tee ./opt-13b-8bit-unstructured50-cali-by-c4/log.log &

# mkdir ./opt-13b-4bit-unstructured50-cali-by-c4
# CUDA_VISIBLE_DEVICES=2 python opt.py facebook/opt-13b c4 --sparsity .5 --wbits 4 --save ./opt-13b-4bit-unstructured50-cali-by-c4 2>&1 | tee ./opt-13b-4bit-unstructured50-cali-by-c4/log.log &

# mkdir ./opt-6.7b-8bit-unstructured50-cali-by-c4
# CUDA_VISIBLE_DEVICES=3 python opt.py facebook/opt-6.7b c4 --sparsity .5 --wbits 8 --save ./opt-6.7b-8bit-unstructured50-cali-by-c4 2>&1 | tee ./opt-6.7b-8bit-unstructured50-cali-by-c4/log.log &

# mkdir ./logs/opt-30b-unstructured50-cali-by-c4
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-30b c4 --sparsity .5  2>&1 | tee ./logs/opt-30b-unstructured50-cali-by-c4/log.log 


# folder=./logs/opt-30b-unstructured50-head10
# mkdir $folder
# CUDA_VISIBLE_DEVICES=0 python opt_lmhead.py facebook/opt-30b c4 --sparsity .5 --lm_head_sparsity .1  2>&1 | tee $folder/log.log &

# folder=./logs/opt-30b-unstructured50-head30
# mkdir $folder
# CUDA_VISIBLE_DEVICES=1 python opt_lmhead.py facebook/opt-30b c4 --sparsity .5 --lm_head_sparsity .3  2>&1 | tee $folder/log.log &

# folder=./logs/opt-30b-unstructured50-head50
# mkdir $folder
# CUDA_VISIBLE_DEVICES=2 python opt_lmhead.py facebook/opt-30b c4 --sparsity .5 --lm_head_sparsity .5  2>&1 | tee $folder/log.log &

# folder=./logs/opt-350m-unstructured50-cali512
# mkdir $folder
# CUDA_VISIBLE_DEVICES=3 python opt_lmhead.py facebook/opt-350m c4 --sparsity .5 --lm_head_sparsity .0 --nsamples 512 2>&1 | tee $folder/log.log &

# folder=./logs/opt-350m-unstructured50-head50-cali2048
# mkdir $folder
# CUDA_VISIBLE_DEVICES=3 python opt_lmhead.py facebook/opt-350m c4 --sparsity .5 --lm_head_sparsity .5  --nsamples 2048 2>&1 | tee $folder/log.log &

# folder=./logs/opt-350m-unstructured50-head30-cali512
# mkdir $folder
# CUDA_VISIBLE_DEVICES=3 python opt_lmhead.py facebook/opt-350m c4 --sparsity .5 --lm_head_sparsity .3  --nsamples 512 2>&1 | tee $folder/log.log &

# folder=./logs/opt-350m-unstructured50-head10-cali512
# mkdir $folder
# CUDA_VISIBLE_DEVICES=3 python opt_lmhead.py facebook/opt-350m c4 --sparsity .5 --lm_head_sparsity .1  --nsamples 512 2>&1 | tee $folder/log.log &


# folder=./logs/opt-30b-unstructured50-head20
# mkdir $folder
# CUDA_VISIBLE_DEVICES=0 python opt_lmhead.py facebook/opt-30b c4 --sparsity .5 --lm_head_sparsity .2  2>&1 | tee $folder/log.log &

# folder=./logs/opt-30b-unstructured50-head40
# mkdir $folder
# CUDA_VISIBLE_DEVICES=1 python opt_lmhead.py facebook/opt-30b c4 --sparsity .5 --lm_head_sparsity .4  2>&1 | tee $folder/log.log &

folder=./logs/opt-30b-unstructured50-head50-cali512
mkdir $folder
CUDA_VISIBLE_DEVICES=2 python opt_lmhead_low_cuda_mem.py facebook/opt-30b c4 --sparsity .5 --lm_head_sparsity .5 --nsamples 512 2>&1 | tee $folder/log.log &

folder=./logs/opt-30b-unstructured50-head50-cali2048
mkdir $folder
CUDA_VISIBLE_DEVICES=3 python opt_lmhead_low_cuda_mem.py facebook/opt-30b c4 --sparsity .5 --lm_head_sparsity .5 --nsamples 2048 2>&1 | tee $folder/log.log &