# Overview
This repository contains an implementation of the CAKE matrix-multiplication algorithm at the network layer for an MPI cluster of heterogenous CPUs. A matrix residing on a server is partitioned into tiles that are scheduled for computation on each host CPU. Each host then uses CAKE to schedule the tile matrix-multiplication on multiple cores. Completed result tiles are aggregated at the server from each host. 

## Installation

Before installing CAKE_on_net, please download and install the folowing dependencies:

* `CAKE_on_CPU (https://github.com/vnatesh/CAKE_on_CPU/)`
* `mpich-3.3.2`

To install CAKE_on_net, simply do:
```bash
git clone https://github.com/vnatesh/CAKE_on_net.git
make
```

## Quick Start

In the `examples` directory, you will find a simple script `cake_sgemm_test.cpp` that performs CAKE matrix multiplication on random input matrices given M, K, and N values as command line arguments. You will need to provide an MPI hostfile containing the hostnames of all nodes involved in your cluster. Make sure you have sourced the `env.sh` file from the [CAKE_on_CPU](https://github.com/vnatesh/CAKE_on_CPU) library on all nodes before running the example. An example hostfile is shown below:

```bash
manager:1
host1:1
host2:1
ubuntuServer:1
alienware:1
```

To compile the script, simple type `make` and run the script using mpiexec as shown below. 

```bash
~/CAKE_on_net/examples$ make
mpic++ -O3 -Wall -I/home/ubuntu/CAKE_on_net/include -I/home/ubuntu/CAKE_on_net/CAKE_on_CPU/include -I/usr/local/include/blis src/cake_sgemm_c2.cpp src/pack_c2.cpp \
src/block_sizing_c2.cpp -L/home/ubuntu/CAKE_on_net/CAKE_on_CPU -lcake -fopenmp -o cake_sgemm_test

~/CAKE_on_net/examples$ mpiexec -f hostfile -np 5  ./cake_sgemm_c2 5000 5000 5000 4

M = 5000, K = 5000, N = 5000
mh = 3344
Hostname: alienware
Host IP: 10.0.0.244
Hostname: ubuntu
Host IP: 10.0.0.184
Hostname: ubuntu
Host IP: 10.0.0.169
Hostname: ubuntu
Host IP: 10.0.0.47

sgemm time: 26.730233 
CORRECT!
```
