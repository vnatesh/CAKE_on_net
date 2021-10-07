#!/bin/bash


# set environment vars for BLIS, CAKE, and mpiP profiling

grep -qxF "export LD_PRELOAD=$PWD/mpiP/libmpiP.so" ~/.bashrc || echo "export LD_PRELOAD=$PWD/mpiP/libmpiP.so" >> ~/.bashrc

grep -qxF 'export MPIP="-c -e -f mpiP_result"' ~/.bashrc || echo 'export MPIP="-c -e -f mpiP_result"' >> ~/.bashrc

grep -qxF 'export BLIS_INSTALL_PATH=/usr/local' ~/.bashrc || echo 'export BLIS_INSTALL_PATH=/usr/local' >> ~/.bashrc

grep -qxF "export CAKE_HOME="$PWD"/CAKE_on_CPU" ~/.bashrc || echo "export CAKE_HOME="$PWD"/CAKE_on_CPU" >> ~/.bashrc

grep -qxF 'LD_LIBRARY_PATH=$CAKE_HOME:$LD_LIBRARY_PATH' ~/.bashrc || echo 'LD_LIBRARY_PATH=$CAKE_HOME:$LD_LIBRARY_PATH' >> ~/.bashrc
