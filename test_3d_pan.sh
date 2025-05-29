# # - - - - - - - -      Testing      - - - - - - - # 

nvidia-smi

##############################################################

# # - - - - - - - - - - - - - - - - - - - - - # 
# #                   Pancrease
# # - - - - - - - - - - - - - - - - - - - - - # 

expname="Pancrease_runs"
version="3"
numlb=6 # 6, 12
gpuid=0

python3 ./code/test_performance_3d.py \
    --root_path /home/chenyu/SSMIS/data/Pancreas \
    --res_path ./results/Pancreas/ \
    --dataset "Pancreas" \
    --gpu ${gpuid} \
    --exp ${expname}/v${version} \
    --labeled_num ${numlb} \
    --model vnet  \
    --model_ext vnet_res18vnet \
    --model_i model1

python3 ./code/test_performance_3d.py \
    --root_path /home/chenyu/SSMIS/data/Pancreas \
    --res_path ./results/Pancreas/ \
    --dataset "Pancreas" \
    --gpu ${gpuid} \
    --exp ${expname}/v${version} \
    --labeled_num ${numlb} \
    --model res18vnet  \
    --model_ext vnet_res18vnet \
    --model_i model2

