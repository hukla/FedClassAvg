# FedClassAvg: Local Representation Learning for Personalized Federated Learning on Heterogeneous Neural Networks

## Requirements
```
mpi4py==3.0.3
torch==1.9.1
```

## Usage example
```
mpirun -hosts [hosts] python src/fedclassavg.py \
--models [resnet,shufflenet,googlenet,alexnet] \
--data [data] \
--mu [mu] \
--data_partition [noniid-labeldir|noniid-twoclass] \
--virtual_per_node [number of virtual nodes] \
--c [client sampling rate] \
--output_path [path to output]
```
