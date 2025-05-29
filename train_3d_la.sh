nvidia-smi

# - - - - - - - - - - - - LA - - - - - - - - - - - - #

expname="LA_runs"
log_dir=./logs/${expname}
mkdir -p ${log_dir}

version="re"
gpuid=2

nohup python3 ./code/train_post_3d_aut_cp.py \
    --gpu_id=${gpuid} \
    --cfg config_3d_la_aut.yml \
    --exp ${expname}/v${version} \
    >${log_dir}/log_v${version}.log 2>&1 &

# python3 ./code/train_post_3d_aut.py \
#     --gpu_id=${gpuid} \
#     --cfg config_3d_la_aut.yml \
#     --exp ${expname}/v${version}
#     >${log_dir}/log_v${version}.log 2>&1 &

