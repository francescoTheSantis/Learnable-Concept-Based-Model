# Compute the metrics associated to the concept representation for the models which uses resnet as a backbone
# Add rows varying the seed to compute the metrics for all the seeds

python compute_concept_metrics.py --seed 0 --backbone "resnet" --dataset "MNIST_even_odd"

#python compute_concept_metrics.py --seed 0 --backbone "resnet" --dataset "MNIST_sum"

#python compute_concept_metrics.py --seed 0 --backbone "resnet" --dataset "CIFAR10"

#python compute_concept_metrics.py --seed 0 --backbone "resnet" --dataset "CIFAR100"

#python compute_concept_metrics.py --seed 0 --backbone "resnet" --dataset "imagenet"

#python compute_concept_metrics.py --seed 0 --backbone "resnet" --dataset "Skin"

#python compute_concept_metrics.py --seed 0 --backbone "resnet" --dataset "CUB200"