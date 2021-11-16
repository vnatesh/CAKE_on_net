#include "cake_c2.h"

// int get_block_dim_C2(unsigned long long dev_dram_sz, double alpha_n, int p) {

//    return (int) sqrt(((double) dev_dram_sz / (4*2*2.0)) / (2 + alpha_n*p));
// }


/*
In each host, we need memory allocated for the following tiles:
   - original A tile, packed A tile (2 mh x kh)
   - original B tile, double buffered B tile, packed B tile (3 mh x alpha*mh)
   - C result tile (mh x alpha*p*mh), double buffered C tile (2 mh x alpha*p*mh)
   - packed C tile for current MM (1 mh x alpha*mh) 
   - use only half of DRAM to account for other processes
*/

int get_block_dim_C2(unsigned long long dev_dram_sz, double alpha_n, int p) {

   return (int) sqrt(((double) dev_dram_sz / (4*2*2.0)) / (1 + 2*alpha_n + alpha_n*p));
}


void init_block_dims_net(int M, int N, int K, int p, blk_dims_net_t* x, int h_max) {

   x->alpha_n = 1.0;
   x->m_h = get_block_dim_C2(1ULL << 27, x->alpha_n, h_max);
   x->k_h = x->m_h;
   x->n_h = (int) (x->alpha_n * p * x->m_h);

   x->p_dev = 4; // number of cores on a single device
   x->mr_dev = 4; //4; // mr on raspbi device
   x->m_r = x->mr_dev*x->p_dev > x->m_h ? x->m_h : x->mr_dev*x->p_dev;

   x->k_pad = (K % x->k_h) ? 1 : 0; 
   x->n_pad = (N % x->n_h) ? 1 : 0; 
   x->m_pad = (M % (p*x->m_h)) ? 1 : 0; 

   x->Mb = (M / (p*x->m_h)) + x->m_pad;
   x->Nb = (N / x->n_h) + x->n_pad;
   x->Kb = (K / x->k_h) + x->k_pad;

   int mr_rem = (int) ceil( ((double) (M % (p*x->m_h))) / x->m_r) ;
   int mr_per_host = (int) ceil( ((double) mr_rem) / p );
   
   if(mr_per_host) 
      x->p_l = (int) ceil( ((double) mr_rem) / mr_per_host);
   else
      x->p_l = 0;

   // int nr_rem = (int) ceil( ((double) (N % n_c) / n_r)) ;
   x->n_h1 = N % x->n_h;

   x->m_h1 = mr_per_host * x->m_r;
   x->m_h1_last_host = (M % (p*x->m_h)) - (x->p_l-1)*x->m_h1;
   // (mr_per_host - (x->p_l*mr_per_host - mr_rem)) * x->m_r;
   x->k_h1 = K % x->k_h;

}

