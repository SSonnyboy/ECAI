nvidia-smi

# - - - - - - - - - - - - - - - - - - - - - - - - #

expname="ACDC_runs"
log_dir=./logs/${expname}
mkdir -p ${log_dir}

version="+re"
gpuid=0

nohup python3 ./code/train_post_2d_aut_cp.py \
    --gpu_id=${gpuid} \
    --cfg config_2d_aut.yml \
    --exp ${expname}/v${version} \
    >${log_dir}/log_v${version}.log 2>&1 &

# python3 ./code/train_post_2d_aut.py \
#     --gpu_id=${gpuid} \
#     --cfg config_2d_aut.yml \
#     --exp ${expname}/v${version}

# - - - - - - - - - - - - - - - - - - - - - - - - #
