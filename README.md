# AutoMC

The source code of "AutoMC: Automated Model Compression Based on Domain Knowledge and Progressive Search".

# Function

AutoMC is an effective and efficient automatic tool for model compression that can automatically design the effective model compression scheme according to the user demands.

## 1. Pretrain

To get pretrained model of VGG-13, VGG-16, VGG-19, ResNet-20, ResNet-56, ResNet-164 on CIFAR-10 and CIFAR-100 (setting pretrained epoch as 200): 

```bash
cd ./CAlgs
python scripts/pretrain.py {0} {1} 200 
# {0} = cifar10 or cifar100
# {1} = vgg13, vgg16, vgg19, resnet20, resnet56 or resnet164
```

IMPOTRTANT: After training the models, copy the model file from `snapshots` folder (eg. `./CAlgs/snapshots/cifar10/vgg13/train/best.finetune.*.pth.tar`) to `trianed_models` folder (eg. `./CAlgs/trianed_models/cifar10/vgg13.pth.tar`).

The trained models also are available to download at: https://drive.google.com/file/d/1t9aSagkk0dbjPmxLnXJBfD1z3T9e6nZH/view?usp=sharing

## 2. Sample Dataset

Sample a mini dataset from the whole dataset.

To sample mini-CIFAR-10 and mini-CIFAR-100 from CIFAR-10 and CIFAR-100 for ResNet-56 and VGG-16 respectively (sample rate is 10%):

```bash
cd ./CAlgs
python generate_mini_dataset.py {0} {1}
# {0} = cifar10 or cifar100
# {1} = vgg16 or resnet56
```

## 3. Pretrain Models on Sampled Dataset

Pretrain models on sampled dataset.

To train VGG-16 and ResNet-56 on mini-CIFAR-10 and mini-CIFAR-100 (setting trained epoch as 50):

```bash
cd ./CAlgs
python scripts/pretrain.py {0} {1} 50
# {0} = mini_cifar10 or mini_cifar100
# {1} = vgg16 or resnet56
```

## 4. Get Results of 6 State-of-the-art Human-invented Compression Methods

We represent LMA, LeGR, NS, SFP, HOS, LFB as 1, 2, 3, 4, 5, 7 in our code.

To get results of 6 human-invented compression methods of VGG-16 and ResNet-56 on CIFAR-10 and CIFAR-100 (setting compression rate as 0.3 and 0.6):

```bash
cd ./CAlgs
python scripts/run.py {0} {1} {2} {3}
# {0} = cifar10 or cifar100
# {1} = vgg16 or resnet56
# {2} = 1, 2, 3, 4, 5, 7
# {3} = 0.3 or 0.6
```

## 5. Get Results of NAS Algorithms

