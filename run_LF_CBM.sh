###### ResNet18 as pre-trained backbone ######
# change the seed to perform multiple runs

# MNIST Even-Odd
python train_LF_CBM.py --seed 0 --num_epochs 40 --batch_size 128 --n_labels 2 --backbone "resnet" --dataset "MNIST_even_odd" --lr 1e-4 --step_size 10 --gamma 0.1 

# MNIST Addition
#python train_LF_CBM.py --seed 0 --num_epochs 40 --batch_size 128 --n_labels 19 --backbone "resnet" --dataset "MNIST_sum" --lr 1e-4 --step_size 10 --gamma 0.1

# CIFAR10
#python train_LF_CBM.py --seed 0 --num_epochs 40 --batch_size 128 --n_labels 10 --backbone "resnet" --dataset "CIFAR10" --lr 1e-4 --step_size 10 --gamma 0.1 

# CIFAR100
#python train_LF_CBM.py --seed 0 --num_epochs 40 --batch_size 128 --n_labels 100 --backbone "resnet" --dataset "CIFAR100" --lr 1e-4 --step_size 10 --gamma 0.1

# Tiny Imagenet
#python train_LF_CBM.py --seed 0 --num_epochs 40 --batch_size 128 --n_labels 200 --backbone "resnet" --dataset "imagenet" --lr 1e-4 --step_size 10 --gamma 0.1

# Skin Lesions
#python train_LF_CBM.py --seed 0 --num_epochs 40 --batch_size 128 --n_labels 4 --backbone "resnet" --dataset "Skin" --lr 1e-4 --step_size 10 --gamma 0.1

#CUB200
#python train_LF_CBM.py --seed 1 --num_epochs 40 --batch_size 128 --n_labels 200 --backbone 'resnet' --dataset "CUB200" --lr 1e-4 --step_size 10 --gamma 0.1 --fine_tune
