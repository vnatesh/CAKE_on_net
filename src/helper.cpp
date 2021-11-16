#include "cake_c2.h"



void launch_gemm_thread(pthread_t* gemm_thread, struct gemm_input* inp) {
#ifdef USE_CAKE
             
   // launch matmul on separate thread to overlap compute with IO
   if(pthread_create(gemm_thread, NULL, cake_sgemm_launch, (void*) inp) != 0) {
      printf("Thread creation failed\n");
   }
#endif

#ifdef USE_MKL
   // launch matmul on separate thread to overlap compute with IO
   if(pthread_create(gemm_thread, NULL, mkl_sgemm_launch, (void*) inp) != 0) {
      printf("Thread creation failed\n");
   }     
#endif

}


// void pack_A_tile(float* A_h, float* A_p, int m_h_t, int n_h_t, int k_h_t, int p_dev, 
//    double alpha, blk_dims_t* xa, cake_cntx_t* cake_cntx) {

// #ifdef USE_CAKE 
//    // pack A and reuse this packed copy for all B tiles in the CB block
//    init_block_dims(m_h_t, n_h_t, k_h_t, p_dev, xa, cake_cntx, KMN);
//    pack_A_single_buf_k_first(A_h, A_p, m_h_t, k_h_t, p_dev, xa, cake_cntx);
// #endif

// #ifdef USE_MKL
//    free(A_p);

//    // allocate memory for packed data format
//    size_t x = cblas_sgemm_pack_get_size(CblasAMatrix, m_h_t, n_h_t, k_h_t);
//    A_p = (float*) mkl_malloc(x, 64 );
//    cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, 
//                   m_h_t, n_h_t, k_h_t, alpha, A_h, k_h_t, A_p);

//  #endif

// }

