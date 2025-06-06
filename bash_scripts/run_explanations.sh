# Run this bash script to generate explanations for the concept attention model on various datasets.
# Add rows varying the seed to compute the metrics for all the seeds

python src/generate_explanations.py --seed 0 --dataset "MNIST_even_odd" --backbone "resnet" --n_samples 10

#python src/generate_explanations.py --seed 0 --dataset "MNIST_sum" --backbone "resnet" --n_samples 10

#python src/generate_explanations.py --seed 0 --dataset "CIFAR10" --backbone "resnet" --n_samples 10

#python src/generate_explanations.py --seed 0 --dataset "CIFAR100" --backbone "resnet" --n_samples 10

#python src/generate_explanations.py --seed 0 --dataset "imagenet" --backbone "resnet" --n_samples 10

#python src/generate_explanations.py --seed 0 --dataset "CUB200" --backbone "resnet" --n_samples 10