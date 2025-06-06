###### ResNet18 as pre-trained backbone ######
# change the seed to perform multiple runs

# MNIST Even-Odd
python src/train_LCBM.py --seed 0 --num_epochs 40 --binarization_step 20 --n_concepts 10 --concept_emb_size 128 --n_labels 2 --deep_parameterization --expand_recon_bottleneck --backbone "resnet" --alpha 0.1 --dataset "MNIST_even_odd" --lr 1e-4 --step_size 10 --gamma 0.1 --lambda_task 1 --lambda_gate 1 --lambda_recon 1 --verbose 10 --bound 20

# MNIST Addition
#python src/train_LCBM.py --seed 0 --num_epochs 40 --binarization_step 20 --n_concepts 10 --concept_emb_size 128 --n_labels 19 --deep_parameterization --expand_recon_bottleneck --backbone "resnet" --alpha 0.2 --dataset "MNIST_sum" --lr 1e-4 --step_size 10 --gamma 0.1 --lambda_task 1 --lambda_gate 1 --lambda_recon 1 --verbose 5 --bound 20

# CIFAR10
#python src/train_LCBM.py --seed 0 --num_epochs 40 --binarization_step 20 --n_concepts 15 --concept_emb_size 128 --n_labels 10 --deep_parameterization --expand_recon_bottleneck --backbone "resnet" --alpha 0.2 --dataset "CIFAR10" --lr 1e-4 --step_size 10 --gamma 0.1 --lambda_task 1 --lambda_gate 1 --lambda_recon 1 --verbose 10 --bound 20

# CIFAR100
#python src/train_LCBM.py --seed 0 --num_epochs 40 --binarization_step 20 --n_concepts 20 --concept_emb_size 128 --n_labels 100 --deep_parameterization --expand_recon_bottleneck --backbone "resnet" --alpha 0.15 --dataset "CIFAR100" --lr 1e-4 --step_size 10 --gamma 0.1 --lambda_task 1 --lambda_gate 1 --lambda_recon 1 --verbose 10 --bound 20

# Tiny Imagenet
#python src/train_LCBM.py --seed 0 --num_epochs 40 --binarization_step 20 --n_concepts 30 --concept_emb_size 128 --n_labels 200 --deep_parameterization --expand_recon_bottleneck --backbone "resnet" --alpha 0.1 --dataset "imagenet" --lr 1e-4 --step_size 10 --gamma 0.1 --lambda_task 1 --lambda_gate 1 --lambda_recon 1 --verbose 10 --bound 20

# Skin Lesions
#python src/train_LCBM.py --seed 0 --num_epochs 40 --binarization_step 20 --n_concepts 14 --concept_emb_size 128 --n_labels 4 --deep_parameterization --expand_recon_bottleneck --backbone "resnet" --alpha 0.1 --dataset "Skin" --lr 1e-4 --step_size 10 --gamma 0.1 --lambda_task 1 --lambda_gate 1 --lambda_recon 1 --verbose 10 --bound 20

# CUB200
#python src/train_LCBM.py --seed 0 --num_epochs 40 --binarization_step 20 --n_concepts 112 --concept_emb_size 128 --n_labels 200 --deep_parameterization --expand_recon_bottleneck --backbone "resnet" --alpha 0.03 --dataset "CUB200" --lr 1e-4 --step_size 10 --gamma 0.1 --lambda_task 1 --lambda_gate 1 --lambda_recon 1 --verbose 10 --bound 20 --fine_tune
