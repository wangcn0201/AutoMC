# AutoMC

The source code of "AutoMC: Automated Model Compression Based on Domain Knowledge and Progressive Search".

# Function

AutoMC is an effective and efficient automatic tool for model compression that can automatically design the effective model compression scheme according to the user demands.

## 1. Pretrain

To get pretrained model of VGG-13, VGG-16, VGG-19, ResNet-20, ResNet-56, ResNet-164 on CIFAR-10 and CIFAR-100 (setting pretrained epoch as 200): 

```bash
cd ./CAlgs
python scripts/pretrain.py cifar10 vgg13 200 
python scripts/pretrain.py cifar10 vgg16 200 
python scripts/pretrain.py cifar10 vgg19 200 
python scripts/pretrain.py cifar10 resnet20 200
python scripts/pretrain.py cifar10 resnet56 200
python scripts/pretrain.py cifar10 resnet164 200

python scripts/pretrain.py cifar100 vgg13 200
python scripts/pretrain.py cifar100 vgg16 200
python scripts/pretrain.py cifar100 vgg19 200
python scripts/pretrain.py cifar100 resnet20 200 
python scripts/pretrain.py cifar100 resnet56 200
python scripts/pretrain.py cifar100 resnet164 200
```

IMPOTRTANT: After training the models, copy the model file from `snapshots` folder (eg. `./CAlgs/snapshots/cifar10/vgg13/train/best.finetune.*.pth.tar`) to `trianed_models` folder (eg. `./CAlgs/trianed_models/cifar10/vgg13.pth.tar`).

The trained models also are available to download at: https://drive.google.com/file/d/1t9aSagkk0dbjPmxLnXJBfD1z3T9e6nZH/view?usp=sharing

## 2. Sample dataset

Sample a mini dataset from the whole dataset.

To sample mini-CIFAR-10 and mini-CIFAR-100 from CIFAR-10 and CIFAR-100 for ResNet-56 and VGG-16 respectively (sample rate is 10%):

```bash
cd ./CAlgs
python generate_mini_dataset.py cifar10 resnet56
python generate_mini_dataset.py cifar100 resnet56
python generate_mini_dataset.py cifar10 vgg16
python generate_mini_dataset.py cifar100 vgg16
```

## 3. Pretrain models on sampled dataset

Pretrain models on sampled dataset.

To train VGG-16 and ResNet-56 on mini-CIFAR-10 and mini-CIFAR-100 (setting trained epoch as 50):

```bash
cd ./CAlgs
python scripts/pretrain.py mini_cifar10 vgg16 50
python scripts/pretrain.py mini_cifar10 resnet56 50
python scripts/pretrain.py mini_cifar100 vgg16 50
python scripts/pretrain.py mini_cifar100 resnet56 50
```

## 4. Get Results of Compression Methods



## 5. Get Results of NAS algorithms

