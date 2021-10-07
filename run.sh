#!/bin/bash


# compile cake_on_net
make;
mkdir mpiP_result;

NTRIALS=1;
NHOSTS=4;
N=10000

# # enable MpiP profiling
# export LD_PRELOAD=$PWD/mpiP/libmpiP.so;
# export MPIP="-c -e -f mpiP_result"

for ((j=1; j <= $NTRIALS; j++));
do
	# for ((i=1; i <= $NHOSTS; i++));
	for ((i=$NHOSTS; i >= 1; i--));
	do
		mpiexec -f hostfile -np $((i+1))  ./cake_sgemm_c2 $N $N $N $i;
		mv mpiP_result/*.mpiP mpiP_result/$i-$j
	done
done



python plots.py $NTRIALS;
