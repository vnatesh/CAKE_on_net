#!/bin/bash

# Install mpiP for communication profiling

sudo apt-get install python-is-python3
sudo apt-get install libunwind-dev
git clone https://github.com/LLNL/mpiP.git
cd mpiP
./configure --with-cc=mpicc --with-cxx=mpic++ --without-f77
make
cd ..


# Install CAKE for CPUs
git clone https://github.com/vnatesh/CAKE_on_CPU.git
cd CAKE_on_CPU
make install
cd ..
./setvars.sh

# install parallel-ssh to simulatenously run scripts on all hosts
sudo apt-get install pssh
echo $'manager\nubuntuServer\nair\nalienware\npro' >> pssh_hosts

# set environment vars for BLIS, CAKE, and mpiP profiling on all hosts
parallel-ssh -i -h pssh_hosts "cd "$PWD"; ./setvars.sh"
