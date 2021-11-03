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

extern int m_h;
extern int k_h;
extern int n_h;
extern int m_h1;
extern int k_h1;
extern int n_h1;
extern int m_h1_last_host;
extern int mr_dev;
extern int m_r;
extern int p_dev;
extern int p_l;
extern int h_max;
extern int m_pad;
extern int k_pad;
extern int n_pad;
extern int Mb;
extern int Kb;
extern int Nb;
extern double alpha_n;


struct result_input {
       float* C;
       MPI_Comm comm;
       int M;
       int N;
};


// void cake_sgemm_net(int M, int N, int K, int p, int taskid, MPI_Comm comm_world);

void init_block_dims(int M, int N, int K, int p, int h_max);

int get_block_dim_C2(unsigned long long dev_dram_sz, double alpha_n, int p);

void pack_A_h(float* A, float* A_p, int M, int K, int m_h, int k_h, int m_r, int p);

void pack_B_h(float* B, float* B_p, int K, int N, int m_h, int k_h, int n_h, double alpha_n, int p);

void unpack_C_h(float* C, float* C_p, int M, int N, int m_h, int n_h, int m_r, double alpha_n, int p);

void cake_sgemm_root(float* A, float* B, float* C, int M, int N, int K, int p, int taskid);

void cake_sgemm_host(int M, int N, int K, int p, int taskid);

void cake_sgemm_net(float* A, float* B, float* C, int M, int N, int K, int p, int taskid);

int run_tests();
