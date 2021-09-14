#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 
#include <math.h>
#include <time.h> 
//#include <omp.h>

#include "blis.h"
#include "mpi.h"



int get_block_dim_C2(unsigned long long dev_dram_sz, double alpha_n, int p);

void pack_A_h(float* A, float* A_p, int M, int K, int m_h, int k_h, int m_r, int p);

void pack_B_h(float* B, float* B_p, int K, int N, int m_h, int k_h, int n_h, double alpha_n, int p);

void unpack_C_h(float* C, float* C_p, int M, int N, int m_h, int n_h, int m_r, double alpha_n, int p);
