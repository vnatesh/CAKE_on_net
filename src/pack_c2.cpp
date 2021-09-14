#include "cake_c2.h"

void pack_A_h(float* A, float* A_p, int M, int K, int m_h, int k_h, int m_r, int p) {

   int k_pad = (K % k_h) ? 1 : 0; 
   int m_pad = (M % (p*m_h)) ? 1 : 0; 
   int Mb = (M / (p*m_h)) + m_pad;
   int Kb = (K / k_h) + k_pad;

   int mr_rem = (int) ceil( ((double) (M % (p*m_h))) / m_r) ;
   int mr_per_host = (int) ceil( ((double) mr_rem) / p );
   int p_l;
   if(mr_per_host) 
      p_l = (int) ceil( ((double) mr_rem) / mr_per_host);
   else
      p_l = 0;

   int m_h1 = mr_per_host * m_r;
   int m_h1_last_host = (M % (p*m_h)) - (p_l-1)*m_h1;
   int k_h1 = K % k_h;

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



void pack_B_h(float* B, float* B_p, int K, int N, int m_h, int k_h, int n_h, double alpha_n, int p) {

   int k_pad = (K % k_h) ? 1 : 0; 
   int n_pad = (N % n_h) ? 1 : 0; 

   int Nb = (N / n_h) + n_pad;
   int Kb = (K / k_h) + k_pad;

   int n_h1 = N % n_h;
   int k_h1 = K % k_h;

   int B_offset = 0;
   int k, n, n_h_t;

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





void unpack_C_h(float* C, float* C_p, int M, int N, int m_h, int n_h, int m_r, double alpha_n, int p) {

   int n_pad = (N % n_h) ? 1 : 0; 
   int m_pad = (M % (p*m_h)) ? 1 : 0; 
   int Mb = (M / (p*m_h)) + m_pad;
   int Nb = (N / n_h) + n_pad;

   int mr_rem = (int) ceil( ((double) (M % (p*m_h))) / m_r) ;
   int mr_per_host = (int) ceil( ((double) mr_rem) / p );
   int p_l;
   if(mr_per_host) 
      p_l = (int) ceil( ((double) mr_rem) / mr_per_host);
   else
      p_l = 0;

   int n_h1 = N % n_h;
   int m_h1 = mr_per_host * m_r;
   int m_h1_last_host = (M % (p*m_h)) - (p_l-1)*m_h1;

   int C_offset = 0;
   int m, n;
   int m_cb, n_h_t, p_used, host;

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
