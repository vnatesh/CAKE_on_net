#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h> 
//#include <omp.h>

#include "blis.h"
#include "mpi.h"
#include "cake.h"



typedef struct blk_dims_net_t {
	int m_h;
	int k_h;
	int n_h;
	int m_h1;
	int k_h1;
	int n_h1;
	int m_h1_last_host;
	int mr_dev;
	int m_r;
	int p_dev;
	int p_l;
	int h_max;
	int m_pad;
	int k_pad;
	int n_pad;
	int Mb;
	int Kb;
	int Nb;
	double alpha_n;
} blk_dims_net_t;


struct result_input {
       float* C;
       MPI_Comm comm;
       int M;
       int N;
};


// void cake_sgemm_net(int M, int N, int K, int p, int taskid, MPI_Comm comm_world);

void init_block_dims_net(int M, int N, int K, int p, blk_dims_net_t* x, int h_max);

int get_block_dim_C2(unsigned long long dev_dram_sz, double alpha_n, int p);

void pack_A_h(float* A, float* A_p, int M, int K, int p, blk_dims_net_t* x);

void pack_B_h(float* B, float* B_p, int K, int N, int p, blk_dims_net_t* x);

void unpack_C_h(float* C, float* C_p, int M, int N, int p, blk_dims_net_t* x);

void cake_sgemm_root(float* A, float* B, float* C, int M, int N, int K, int p, blk_dims_net_t* x, int taskid);

void cake_sgemm_host(int M, int N, int K, int p, blk_dims_net_t* x, int taskid);

void cake_sgemm_net(float* A, float* B, float* C, int M, int N, int K, int p, int taskid);

int run_tests();
