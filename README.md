# FedClassAvg: Local Representation Learning for Personalized Federated Learning on Heterogeneous Neural Networks

## Requirements
```
matplotlib==3.5.1
mpi4py==3.0.3
numpy==1.21.5
scikit-learn-intelex==2021.5.3
scikit-learn==1.0.2
scipy==1.7.3
seaborn==0.11.2
torch==1.11.0
torchvision==0.12.0
tqdm==4.64.0
wandb==0.12.14
```

## Usage example
```
# Training CIFAR-10 with heterogeneous models with Dir(0.5) non-IID data distribution
mpirun -hosts hostsfile python src/fedclassavg.py \
--models resnet,shufflenet,googlenet,alexnet \
--data cifar10 \
--mu 0.1 \
--data_partition noniid-labeldir \
--c 1 \
--output_path path_to_output] \
| tee path_to_log # save output logs for evaluation

# Calculate average test accuracies among clients
python src/evaluate.py --log_path path_to_logfile --eval_round round_to_evaluate
```
**NOTE**: The exact numbers in experimental results may slightly vary from run to run in different environments because of inherent randomness and nondeterministic implementations in PyTorch. Relative performance of the different algorithms compared remains the same.