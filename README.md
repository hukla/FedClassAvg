# FedClassAvg: Local Representation Learning for Personalized Federated Learning on Heterogeneous Neural Networks

## Requirements
```
mpi4py==3.0.3
torch==1.9.1
```

## Usage example
```
# Training CIFAR-10 with heterogeneous models with Dir(0.5) non-IID data distribution
mpirun -hosts [hosts] python src/fedclassavg.py \
--models resnet,shufflenet,googlenet,alexnet \
--data cifar10 \
--mu 0.1 \
--data_partition noniid-labeldir \
--c 1 \
--output_path [path to output] \
--seed 2022
| tee [path to logfile] # save output logs for evaluation

# evaluate
python src/evaluate.py \
--log_path [path to logfile] \
--eval_round [round to evaluate]
```