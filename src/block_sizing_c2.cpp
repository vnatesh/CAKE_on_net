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


void init_block_dims(int* mh, int* kh, int* nh,
                     int* mh1, int* kh1, int* nh1,
                     int* mh1_last_host, int* mrdev, int* mr, 
                     int* pdev, int* pl, double* alphan,
                     int* mpad, int* kpad, int* npad,
                     int* Mb, int* Kb, int* Nb,
                     int M, int N, int K, int p, int h_max) {

   int m_h, k_h, n_h;
   double alpha_n = 1.0;
   m_h = get_block_dim_C2(1ULL << 27, alpha_n, h_max);
   k_h = m_h;
   n_h = (int) (alpha_n * p * m_h);

   int p_dev = 4; // number of cores on a single device
   int mr_dev = 4; //4; // mr on raspbi device
   int m_r = mr_dev*p_dev > m_h ? m_h : mr_dev*p_dev;

   int k_pad = (K % k_h) ? 1 : 0; 
   int n_pad = (N % n_h) ? 1 : 0; 
   int m_pad = (M % (p*m_h)) ? 1 : 0; 

   int M_b = (M / (p*m_h)) + m_pad;
   int N_b = (N / n_h) + n_pad;
   int K_b = (K / k_h) + k_pad;

   int mr_rem = (int) ceil( ((double) (M % (p*m_h))) / m_r) ;
   int mr_per_host = (int) ceil( ((double) mr_rem) / p );
   int p_l;
   if(mr_per_host) 
      p_l = (int) ceil( ((double) mr_rem) / mr_per_host);
   else
      p_l = 0;

   // int nr_rem = (int) ceil( ((double) (N % n_c) / n_r)) ;
   int n_h1 = N % n_h;

   int m_h1 = mr_per_host * m_r;
   int m_h1_last_host = (M % (p*m_h)) - (p_l-1)*m_h1;
   // (mr_per_host - (p_l*mr_per_host - mr_rem)) * m_r;
   int k_h1 = K % k_h;

   *mh = m_h; *kh = k_h; *nh = n_h;
   *mh1 = m_h1; *kh1 = k_h1; *nh1 = n_h1;
   *mh1_last_host = m_h1_last_host; *mrdev = mr_dev; *mr = m_r; 
   *pdev = p_dev; *pl = p_l; *alphan = alpha_n;
   *mpad = m_pad; *kpad = k_pad; *npad = n_pad;
   *Mb = M_b; *Kb = K_b; *Nb = N_b;
}

