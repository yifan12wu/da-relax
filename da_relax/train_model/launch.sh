#!/bin/bash

#python -u -B run.py --dataset=cifar10 --model=resnet18 --config_file=resnet_mm_untuned
#python -u -B run.py --dataset=cifar5 --model=resnet18 --config_file=resnet_mm_untuned
#python -u -B run.py --dataset=svhn --model=resnet18 --config_file=resnet_mm_untuned
#python -u -B run.py --dataset=svhn5 --model=resnet18 --config_file=resnet_mm_untuned

python -u -B run.py --dataset=cifar10 --model=lenet --config_file=lenet_mm
python -u -B run.py --dataset=cifar5 --model=lenet --config_file=lenet_mm
python -u -B run.py --dataset=svhn --model=lenet --config_file=lenet_mm
python -u -B run.py --dataset=svhn5 --model=lenet --config_file=lenet_mm
