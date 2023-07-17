#!/bin/bash
# specify the device id pytorch can see, eg. "0,1,2,3"
cuda_devices=$1
echo "Set environ CUDA devices $cuda_devices visible"
num_device=$2
# specify the number of devices pytorch will use, the number must not be more than the number of devices initialized above
echo "Using $num_device CUDA devices for training"
exp_name=$3
echo "Experiment name: $exp_name"
other_args=$4
echo "Additional running args: $other_args"
CUDA_VISIBLE_DEVICES="$cuda_devices" torchrun --nproc_per_node=$num_device \
--rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 train.py \
--mixed --benchmark --exp_name $exp_name --wandb $other_args> logs/${exp_name}.log &