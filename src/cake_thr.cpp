#include "cake_c2.h"


void* cake_sgemm_launch(void* inputs) {
	
    struct gemm_input* inp = (struct gemm_input*) inputs;    
    double ans = cake_sgemm(inp->A, inp->B, inp->C, inp->M, inp->N, inp->K, inp->p, 
    	inp->cake_cntx, inp->packedA, inp->packedB, inp->alpha, inp->beta);
	//sleep(4);
	pthread_exit(NULL);
}
