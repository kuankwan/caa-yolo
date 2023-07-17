#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=295001 ./tools/dist_train.sh configs/custom/yolov8_s_sim_city_ours.py 4
#CUDA_VISIBLE_DEVICES=6 PORT=295001 ./tools/dist_train.sh configs/custom/yolov8_s_city_foggy_ours_mt.py 1
CONFIG=$1
GPUS=$1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-295002}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
CUDA_VISIBLE_DEVICES="7"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
