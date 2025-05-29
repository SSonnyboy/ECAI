nvidia-smi

# - - - - - - - -     Testing      - - - - - - - # 
expname="ACDC_runs"
version="+re"
numlb=3 # 3, 7, 14
gpuid=0

python3 ./code/test_performance_2d.py \
    --root_path /home/chenyu/SSMIS/data/ACDC \
    --res_path ./results/ACDC \
    --gpu_id=${gpuid} \
    --exp ${expname}/v${version} \
    --labeled_num ${numlb} \
    --model res18unet  \
    --model_ext unet_res18unet \
    --model_i model2
# python3 ./code/test_performance_2d.py \
#     --root_path /home/chenyu/SSMIS/data/ACDC \
#     --res_path ./results/ACDC \
#     --gpu_id=${gpuid} \
#     --exp ${expname}/v${version} \
#     --labeled_num ${numlb} \
#     --model res18unet  \
#     --model_ext unet_res18unet \
#     --model_i model2
