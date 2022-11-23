# MATCHA Reproducing and Cleaning Code

In this repo, we alter the code from the MATCHA: Communication-Efficient Decentralized SGD code repository (CODE: https://github.com/JYWa/MATCHA, PAPER: https://arxiv.org/abs/1905.09435). Furthermore, we reproduce their results training Resnet to classify Cifar10 using a Decentralized Communicator. Several components of the original code needed serious modification to run correctly in UMIACS. The original overhaul was done by Tahseen Rabbani and Marco Bornstein. Rabbani added differential privacy functionality to this repo. 

## Dependencies

Our code was run on the University of Maryland Center for Machine Learning (CML) cluster (https://wiki.umiacs.umd.edu/umiacs/index.php/CML).


You will need gcc >= 8.1.0, cuda >= 11.1.1, openmpi, and python >=3.9. 
On the UMIACS CML cluster you can use:
```
$ module load python3
$ module load gcc/8.1.0
$ module load openmpi
$ module switch cuda/11.1.1
```
I highly recommend you do this in a virtual env. 

## Running the Code

Modify run.sh as necessary (see below) and then use
```
$sbatch run.sh
```
******On your very first run set --downloadCifar 1 to pull in the Cifar10 dataset********
We provide examples for how to modify the run.sh command line to accommodate various decentralized experiments. 

######Vanilla decenSGD######## 
Example: noise=0.3, 8 workers, 200 epochs:
```
mpirun -np 8 python matcha-train.py --description vanNoise0.3 --resSize 50 --randomSeed 9001 --datasetRoot ./data --budget 1 --outputFolder vanNoise0.3Output --bs 64 --epoch 200 --noise 0.3 --action_store False --name vanNoise0.3
```
You would change the noise flag to simulate the various graphs in my paper. 

####MATCHA######
Example: noise=0.3, 8 workers, 200 epochs, budget=0.5: 
```
mpirun -np 8 python matcha-train.py --description Matcha0.3 --resSize 50 --randomSeed 9001 --datasetRoot ./data --budget 0.5 --outputFolder Matcha0.3Output --bs 64 --epoch 200 --noise 0.3 --action_store False --name Matcha0.3
```
You would change the noise flag to simulate the various graphs in my paper. 

####P-DSGD#######
Example: noise=0.3, 8 workers, 200 epochs, budget=0.5: 
```
mpirun -np 8 python train_mpi.py --description Matcha0.3 --resSize 50 --randomSeed 9001 --datasetRoot ./data --budget 0.5 --outputFolder Matcha0.3Output --bs 64 --epoch 200 --noise 0.3 --name Matcha0.3
```
You would change the noise flag to simulate the various graphs in my paper. 

These experiments take a long time. On the order of 2 hours each even with 2 GPUs. 