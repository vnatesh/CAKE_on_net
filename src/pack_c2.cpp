#include "cake_c2.h"

void pack_A_h(float* A, float* A_p, int M, int K, int p, blk_dims_net_t* x) {

   int m_h = x->m_h, k_h = x->k_h;
   int m_h1 = x->m_h1, k_h1 = x->k_h1;
   int m_h1_last_host = x->m_h1_last_host;
   int p_l = x->p_l;
   int m_pad = x->m_pad, k_pad = x->k_pad;
   int Mb = x->Mb, Kb = x->Kb;


// printf("packing %d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n\n", 
//    m_h,k_h,m_h1,k_h1,m_h1_last_host,p_l,m_pad,k_pad,Mb,Kb);


   int m, k, A_offset = 0;
   int m_cb, k_h_t, p_used, host;

   for(m = 0; m < Mb; m++) {

      if((m == Mb - 1) && m_pad) {
         p_used = p_l;
         m_cb = M % (p*m_h);
      } else {
         p_used = p;
         m_cb = p_used*m_h;
      }

      for(k = 0; k < Kb; k++) {
         
         k_h_t = k_h; 
         if((k == Kb - 1) && k_pad) {
            k_h_t = k_h1;
         }

         #pragma omp parallel for private(host)
         for(host = 0; host < p_used; host++) {

            int m_h_t, m_h_x;

            if((m == Mb - 1) && m_pad) {
               m_h_t = (host == (p_l - 1) ? m_h1_last_host : m_h1);
               m_h_x = m_h1;
            } else {
               m_h_t = m_h;
               m_h_x = m_h;
            }

            for(int i = 0; i < m_h_t; i++) {
               for(int j = 0; j < k_h_t; j++) {
                  A_p[A_offset + host*m_h_x*k_h_t + i*k_h_t + j] = 
                  A[m*p*m_h*K + k*k_h + host*m_h_x*K + i*K + j];
               }
            }
         }

         A_offset += m_cb*k_h_t;
      }
   }
}



void pack_B_h(float* B, float* B_p, int K, int N, int p, blk_dims_net_t* x) {

   int m_h = x->m_h, k_h = x->k_h, n_h = x->n_h;
   int k_h1 = x->k_h1, n_h1 = x->n_h1;
   int k_pad = x->k_pad, n_pad = x->n_pad;
   int Kb = x->Kb, Nb = x->Nb;
   double alpha_n = x->alpha_n;

   int B_offset = 0;
   int k, n, n_h_t;


printf("packing %d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n\n", 
   m_h,k_h,n_h,k_h1,n_h1,k_pad,n_pad,Kb,Nb);


   for(n = 0; n < Nb; n++) {

      n_h_t = n_h;
      if((n == Nb - 1) && n_pad) {
         n_h_t = n_h1;
      }

      #pragma omp parallel for private(k)
      for(k = 0; k < Kb; k++) {
         
         int k_h_t = k_h; 
         if((k == Kb - 1) && k_pad) {
            k_h_t = k_h1;
         }

         int z1 = (int) (alpha_n*m_h);
         int num_B = (n_h_t / z1) + ((n_h_t % z1) ? 1 : 0);
         int n_rem = n_h_t % z1;
         for(int b = 0; b < num_B; b++) {
            
            int n_pc = z1;
            if((b == num_B - 1) && (n_rem)) {
               n_pc = n_rem;
            }

            for(int i = 0; i < k_h_t; i++) {
               for(int j = 0; j < n_pc; j++) {
                  B_p[B_offset + k*k_h*n_h_t + b*z1*k_h_t + i*n_pc + j] = 
                  B[n*n_h + k*k_h*N + b*z1 + i*N + j];
               }
            }
         }
      }

      B_offset += n_h_t*K;
   }
}





void unpack_C_h(float* C, float* C_p, int M, int N, int p, blk_dims_net_t* x) {

   int m_h = x->m_h, n_h = x->n_h;
   int m_h1 = x->m_h1, n_h1 = x->n_h1;
   int m_h1_last_host = x->m_h1_last_host;
   int p_l = x->p_l;
   int m_pad = x->m_pad, n_pad = x->n_pad;
   int Mb = x->Mb, Nb = x->Nb;
   double alpha_n = x->alpha_n;

   int C_offset = 0;
   int m, n;
   int m_cb, n_h_t, p_used, host;

// printf("unpackng %d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%f\n\n", 
//    m_h,n_h,m_h1,n_h1,m_h1_last_host,p_l,m_pad,n_pad,Mb,Nb,alpha_n);


   for(n = 0; n < Nb; n++) {

      n_h_t = n_h;
      if((n == Nb - 1) && n_pad) {
         n_h_t = n_h1;
      }

      for(m = 0; m < Mb; m++) {

         if((m == Mb - 1) && m_pad) {
            p_used = p_l;
            m_cb = M % (p*m_h);
         } else {
            p_used = p;
            m_cb = p_used*m_h;
         }

         #pragma omp parallel for private(host)
         for(host = 0; host < p_used; host++) {

            int m_h_t, m_h_x;

            if((m == Mb - 1) && m_pad) {
               m_h_t = (host == (p_l - 1) ? m_h1_last_host : m_h1);
               m_h_x = m_h1;
            } else {
               m_h_t = m_h;
               m_h_x = m_h;
            }

            int z1 = (int) (alpha_n*m_h);
            int num_B = (n_h_t / z1) + ((n_h_t % z1) ? 1 : 0);
            int n_rem = n_h_t % z1;
            for(int b = 0; b < num_B; b++) {
               
               int n_pc = z1;
               if((b == num_B - 1) && (n_rem)) {
                  n_pc = n_rem;
               }

               for(int i = 0; i < m_h_t; i++) {
                  for(int j = 0; j < n_pc; j++) {
                     C[n*n_h + m*p*m_h*N + host*m_h_x*N + b*z1 + i*N + j] = 
                     C_p[C_offset + host*m_h_x*n_h_t + b*m_h_t*z1 + i*n_pc + j];
                     // printf("(%d,%d,%d,%d,%d,%d,%d,%d,%d)  ", 
                        // C_offset,host,m_h_x,n_h_t,b,z1,i,n_pc,j);
                     // printf("%d ", n*n_h + m*p*m_h*N + host*m_h_x*N + b*z1 + i*N + j);
                  }
               }
            }
         }
         
         C_offset += m_cb*n_h_t;
      }
   }
}
