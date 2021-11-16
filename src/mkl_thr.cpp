#include "cake_c2.h"

void* mkl_sgemm_launch(void* inputs) {

	struct gemm_input* inp = (struct gemm_input*) inputs;    
	mkl_set_num_threads(inp->p);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            inp->M, inp->N, inp->K, inp->alpha, inp->A, 
	            inp->K, inp->B, inp->N, inp->beta, inp->C, inp->N);

	pthread_exit(NULL);
}


void* mkl_packed_sgemm_launch(void* inputs) {
   
	struct gemm_input* inp = (struct gemm_input*) inputs;    

	mkl_set_num_threads(inp->p);

	// SGEMM computations are performed using the packed A matrix Ap
	cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans,
	            inp->M, inp->N, inp->K, inp->A, inp->K, inp->B, 
	            inp->N, inp->beta, inp->C, inp->N);
	// free(inp->A);

   pthread_exit(NULL);
}

