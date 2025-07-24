
TRIAL=${1}
NET=${2}
mkdir checkpoints
mkdir checkpoints/${NET}_${TRIAL}
torchrun --nproc_per_node=1 --master_port 29502 ./train.py --use_gpu --net ${NET} --name ${NET}_${TRIAL}
torchrun --nproc_per_node=1 --master_port 29502 ./test_dataset_model.py --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}/latest_net_.pth
