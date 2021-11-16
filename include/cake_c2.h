#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#include <omp.h>
#include <pthread.h>
#include "mpi.h"



#ifdef USE_MKL
#include "mkl.h"
#define DEBUG 1
#define CHECK_PRINT 0

typedef struct cake_cntx_t {
} cake_cntx_t;

typedef struct blk_dims_t {
} blk_dims_t;

void rand_init(float* mat, int r, int c) ;
bool cake_sgemm_checker(float* A, float* B, float* C, int N, int M, int K) ;
void* mkl_sgemm_launch(void* inputs);
void* mkl_packed_sgemm_launch(void* inputs) ;

#endif


#ifdef USE_CAKE
#include "cake.h"

void* cake_sgemm_launch(void* inputs);

#endif




struct gemm_input {
	float* A; 
	float* B;
	float* C;
	float alpha;
	float beta;
	int M;
	int N;
	int K; 
	int p;
	cake_cntx_t* cake_cntx;
	bool packedA;
	bool packedB;
};


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

void launch_gemm_thread(pthread_t* gemm_thread, struct gemm_input* inp) ;

void pack_A_tile(float* A_h, float* A_p, int m_h_t, int n_h_t, int k_h_t, int p_dev, 
   double alpha, blk_dims_t* xa, cake_cntx_t* cake_cntx);
// int run_tests();
