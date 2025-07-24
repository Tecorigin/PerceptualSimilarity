
TRIAL=${1}
NET=${2}
mkdir checkpoints
mkdir checkpoints/${NET}_${TRIAL}_tune
torchrun --nproc_per_node=1 --master_port 29503 ./train.py --train_trunk --use_gpu --net ${NET} --name ${NET}_${TRIAL}_tune
torchrun --nproc_per_node=1 --master_port 29503 ./test_dataset_model.py --train_trunk --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}_tune/latest_net_.pth
