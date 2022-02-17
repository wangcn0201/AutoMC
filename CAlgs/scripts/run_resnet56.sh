'''
压缩任务：模型（resnet【56】、vgg【16】}）+数据集（cifar10、cifar100）+压缩率（0.3、0.6）
压缩任务：2*2*2=8个
压缩算法：6个【C1、C2、C3、C4、C5、C7】+4个【NAS】
'''

CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py cifar10 resnet56 1 0.6 > ./logs/log_run_cifar10_resnet56_1_0.6.log &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py cifar10 resnet56 3 0.6 > ./logs/log_run_cifar10_resnet56_3_0.6.log &
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py cifar10 resnet56 4 0.6 > ./logs/log_run_cifar10_resnet56_4_0.6.log &
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py cifar10 resnet56 5 0.6 > ./logs/log_run_cifar10_resnet56_5_0.6.log &
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py cifar10 resnet56 7 0.6 > ./logs/log_run_cifar10_resnet56_7_0.6.log &
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py cifar10 resnet56 2 0.6 > ./logs/log_run_cifar10_resnet56_2_0.6.log &

CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py cifar100 resnet56 1 0.6 > ./logs/log_run_cifar100_resnet56_1_0.6.log &
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py cifar100 resnet56 3 0.6 > ./logs/log_run_cifar100_resnet56_3_0.6.log &
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py cifar100 resnet56 4 0.6 > ./logs/log_run_cifar100_resnet56_4_0.6.log &
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py cifar100 resnet56 5 0.6 > ./logs/log_run_cifar100_resnet56_5_0.6.log &
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py cifar100 resnet56 7 0.6 > ./logs/log_run_cifar100_resnet56_7_0.6.log &
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py cifar100 resnet56 2 0.6 > ./logs/log_run_cifar100_resnet56_2_0.6.log &

CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py cifar10 resnet56 1 0.3 > ./logs/log_run_cifar10_resnet56_1_0.3.log &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py cifar10 resnet56 3 0.3 > ./logs/log_run_cifar10_resnet56_3_0.3.log &
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py cifar10 resnet56 4 0.3 > ./logs/log_run_cifar10_resnet56_4_0.3.log &
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py cifar10 resnet56 5 0.3 > ./logs/log_run_cifar10_resnet56_5_0.3.log &
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py cifar10 resnet56 7 0.3 > ./logs/log_run_cifar10_resnet56_7_0.3.log &
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py cifar10 resnet56 2 0.3 > ./logs/log_run_cifar10_resnet56_2_0.3.log &

CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py cifar100 resnet56 1 0.3 > ./logs/log_run_cifar100_resnet56_1_0.3.log &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py cifar100 resnet56 3 0.3 > ./logs/log_run_cifar100_resnet56_3_0.3.log &
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py cifar100 resnet56 4 0.3 > ./logs/log_run_cifar100_resnet56_4_0.3.log &
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py cifar100 resnet56 5 0.3 > ./logs/log_run_cifar100_resnet56_5_0.3.log &
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py cifar100 resnet56 7 0.3 > ./logs/log_run_cifar100_resnet56_7_0.3.log &
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py cifar100 resnet56 2 0.3 > ./logs/log_run_cifar100_resnet56_2_0.3.log &