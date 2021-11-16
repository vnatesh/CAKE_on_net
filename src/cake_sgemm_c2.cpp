#include "cake_c2.h"



// TODO...programmatically get mr/nr for intel skylake correctly
// TODO...get h_max from sbatch script input
// TODO... add input for DRAM size of node to init_block_dims_net
void cake_sgemm_net(float* A, float* B, float* C, int M, int N, int K, int p, int taskid) {

   int mat_inputs[4];
   int h_max = 4;
   blk_dims_net_t* blk_dims_net = (blk_dims_net_t*) malloc(sizeof(blk_dims_net_t));
   init_block_dims_net(M, N, K, p, blk_dims_net, h_max);

   if(taskid == 0) {
      
      struct timespec start, end;
      double diff_t;
      long seconds, nanoseconds;

      mat_inputs[0] = M;
      mat_inputs[1] = K;
      mat_inputs[2] = N;
      mat_inputs[3] = p;
      // bcast M,N,K,p
      MPI_Bcast(mat_inputs, 4, MPI_INT, 0, MPI_COMM_WORLD);

      clock_gettime(CLOCK_REALTIME, &start);

      cake_sgemm_root(A, B, C, M, N, K, p, blk_dims_net, taskid);

      clock_gettime(CLOCK_REALTIME, &end);
      seconds = end.tv_sec - start.tv_sec;
      nanoseconds = end.tv_nsec - start.tv_nsec;
      diff_t = seconds + nanoseconds*1e-9;
      printf("sgemm time: %f \n", diff_t); 

      cake_sgemm_checker(A, B, C, N, M, K);

      free(A);
      free(B);
      free(C);      
   }

   // hosts bcast recv M,N,K,p from server
   else {

      MPI_Bcast(mat_inputs, 4, MPI_INT, 0, MPI_COMM_WORLD);
      M = mat_inputs[0];
      K = mat_inputs[1];
      N = mat_inputs[2];
      p = mat_inputs[3];

      cake_sgemm_host(M, N, K, p, blk_dims_net, taskid);
   }
}


void cake_sgemm_root(float* A, float* B, float* C, int M, int N, int K, int p, blk_dims_net_t* x, int taskid) {

   int m_h = x->m_h, k_h = x->k_h, n_h = x->n_h;
   int m_h1 = x->m_h1, k_h1 = x->k_h1, n_h1 = x->n_h1;
   int m_h1_last_host = x->m_h1_last_host;
   int p_l = x->p_l;
   int m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
   int Mb = x->Mb, Kb = x->Kb, Nb = x->Nb;
   double alpha_n = x->alpha_n;

   // creates an extra communicator for p_l hosts involved in computing the m-padded region
   int color = taskid <= p_l ? 1 : MPI_UNDEFINED; // Determine color based on rank and p_l 
   MPI_Comm comm_pad, comm_used;
   MPI_Comm_split(MPI_COMM_WORLD, color, taskid, &comm_pad);

   printf("M = %d, K = %d, N = %d\n", M,K,N);
   printf("mh = %d\n", m_h);

   omp_set_num_threads(10); // TODO: use 10 threads only on intel i9 10900K

   // copy A,B,C for packing
   float* A_p = (float*) malloc(M * K * sizeof( float ));
   float* B_p = (float*) malloc(K * N * sizeof( float ));
   float* C_p = (float*) calloc(M * N , sizeof( float ));

   // pack A and B before sending to hosts
   pack_A_h(A, A_p, M, K, p, x);
   pack_B_h(B, B_p, K, N, p, x);

   if(DEBUG) printf("m_h = %d, k_h = %d, n_h = %d\n", m_h, k_h, n_h);

   int A_offset = 0;
   int B_offset = 0;
   int C_offset = 0;
   int* sendcounts = (int*) calloc((p + 1) , sizeof(int));
   int* displs_A = (int*) calloc((p + 1) , sizeof(int));
   int* displs_C = (int*) calloc((p + 1) , sizeof(int));
   int* recvcounts = (int*) calloc((p + 1) , sizeof(int));

   int m, k, n;
   int m_cb, n_h_t, k_h_t, p_used, host;


   MPI_Request req;
   MPI_Status status;


   for(n = 0; n < Nb; n++) {

      n_h_t = n_h;
      if((n == Nb - 1) && n_pad) {
         n_h_t = n_h1;
      }

      A_offset = 0; 

      for(m = 0; m < Mb; m++) {

         if((m == Mb - 1) && m_pad) {
            p_used = p_l;
            m_cb = M % (p*m_h);
            comm_used = comm_pad;
            // comm_result_used = comm_result_pad;
         } else {
            p_used = p;
            m_cb = p_used*m_h;
            comm_used = MPI_COMM_WORLD;
            // comm_result_used = comm_result;
         }

         B_offset = n*K*n_h;

         for(k = 0; k < Kb; k++) {
         
            k_h_t = k_h; 
            if((k == Kb - 1) && k_pad) {
               k_h_t = k_h1;
            }

            // scatterv A among p_used procs
            int curr_disp = 0;

            for(host = 0; host < p_used; host++) {

               if((m == Mb - 1) && m_pad) {
                  sendcounts[host+1] = (host == (p_l - 1) ? m_h1_last_host*k_h_t : m_h1*k_h_t);
               } else {
                  sendcounts[host+1] = m_h*k_h_t;
               }

               displs_A[host+1] = curr_disp;
               curr_disp += sendcounts[host+1];
            }

            MPI_Scatterv(&A_p[A_offset], sendcounts, displs_A,
                  MPI_FLOAT, NULL, 0, MPI_FLOAT, 0, comm_used);

            A_offset += m_cb*k_h_t;
            memset(sendcounts, 0, (p + 1) * sizeof(int));
            memset(displs_A, 0, (p + 1) * sizeof(int));


            int z1 = (int) (alpha_n*m_h);
            int num_B = (n_h_t / z1) + ((n_h_t % z1) ? 1 : 0);
            int n_rem = n_h_t % z1;
            for(int i = 0; i < num_B; i++) {
               // bcast B piece among p_used procs
               int bcast_cnt = k_h_t * z1;
               if((i == num_B - 1) && (n_rem)) {
                  bcast_cnt = k_h_t * n_rem;
               }

               MPI_Bcast(&B_p[B_offset], bcast_cnt, MPI_FLOAT, 0, comm_used);
               B_offset += bcast_cnt;
            }

         }


         if(C_offset) {
            MPI_Wait(&req, &status);
            memset(recvcounts, 0, (p + 1)*sizeof(int));
            memset(displs_C, 0, (p + 1)*sizeof(int));
         } 

         // // gatherv C
         int curr_disp = 0;
         for(host = 0; host < p_used; host++) {

            if((m == Mb - 1) && m_pad) {
               recvcounts[host+1] = (host == (p_l - 1) ? m_h1_last_host*n_h_t : m_h1*n_h_t);
            } else {
               recvcounts[host+1] = m_h*n_h_t;
            }

            displs_C[host+1] = curr_disp;
            curr_disp += recvcounts[host+1];
         }

         MPI_Igatherv(NULL, 0, MPI_FLOAT, &C_p[C_offset], recvcounts, displs_C,
            MPI_FLOAT, 0, comm_used, &req);


         C_offset += m_cb*n_h_t;

      }
   }

   // pthread_join(thr_result_server, NULL);
   MPI_Wait(&req, &status);

   unpack_C_h(C, C_p, M, N, p, x);

   free(sendcounts);
   free(recvcounts);
   free(displs_A);
   free(displs_C);
   free(A_p);
   free(B_p);
   free(C_p);
}



void cake_sgemm_host(int M, int N, int K, int p, blk_dims_net_t* x, int taskid) {

   x->h_max = 4;


   int m_h = x->m_h, k_h = x->k_h, n_h = x->n_h;
   int m_h1 = x->m_h1, k_h1 = x->k_h1, n_h1 = x->n_h1;
   int m_h1_last_host = x->m_h1_last_host;
   int mr_dev = x->mr_dev;
   int p_dev = x->p_dev, p_l = x->p_l;
   int m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
   int Mb = x->Mb, Kb = x->Kb, Nb = x->Nb;
   double alpha_n = x->alpha_n;

   // creates an extra communicator for p_l hosts involved in computing the m-padded region
   int color = taskid <= p_l ? 1 : MPI_UNDEFINED; // Determine color based on rank and p_l 
   MPI_Comm comm_pad, comm_used;
   MPI_Comm_split(MPI_COMM_WORLD, color, taskid, &comm_pad);

   omp_set_num_threads(p_dev);

   char hostbuffer[256];
   char *IPbuffer;
   struct hostent *host_entry;
   // int hostname;

   // To retrieve hostname
   gethostname(hostbuffer, sizeof(hostbuffer));
   host_entry = gethostbyname(hostbuffer);

   IPbuffer = inet_ntoa(*((struct in_addr*)
               host_entry->h_addr_list[0]));

   printf("Hostname: %s\n", hostbuffer);
   printf("Host IP: %s\n", IPbuffer);

   int host = taskid - 1;

   float *A_h, *A_p, *B_h, *B_h1, *B_h2, *C_h, *C_h1 = NULL, *C_h2 = NULL;

   int m = 0, k = 0, n = 0;
   int nb_ind = 0;
   int n_h_t, k_h_t, m_h_t;

   printf("HEYY %d\n",host);

   if(posix_memalign((void**) &A_h, 64, m_h * k_h * sizeof(float))) {
      printf("posix memalign error\n");
      exit(1);
   }

   if(posix_memalign((void**) &A_p, 64, (m_h+mr_dev) * k_h * sizeof(float))) {
      printf("posix memalign error\n");
      exit(1);
   }

   // double buffer for B_h
   if(posix_memalign((void**) &B_h1, 64, k_h * n_h * sizeof(float))) {
      printf("posix memalign error\n");
      exit(1);
   }

   if(posix_memalign((void**) &B_h2, 64, k_h * n_h * sizeof(float))) {
      printf("posix memalign error\n");
      exit(1);
   }

   // asynchronous thread to launch cake_sgemm to overlap IO and compute
   pthread_t gemm_thread;

   struct gemm_input* inp = (struct gemm_input*) malloc(sizeof(gemm_input));
   inp->packedA = 1;
   inp->packedB = 0;
   inp->alpha = 1;
   inp->beta = 1;

   bool started = 0;
   bool buf_C = 1;
   MPI_Request req;
   MPI_Status status;

   blk_dims_t* xa = (blk_dims_t*) malloc(sizeof(blk_dims_t));
   cake_cntx_t* cake_cntx = cake_query_cntx();

   while(n < Nb) {

      if((m == Mb - 1) && m_pad && (host >= p_l)) {
         k = 0;
         m = 0;
         n++;
      }

      else {

         if((m == Mb - 1) && m_pad) {
            m_h_t = (host == (p_l - 1) ? m_h1_last_host : m_h1);
            comm_used = comm_pad;
         } else {
            m_h_t = m_h;
            comm_used = MPI_COMM_WORLD;
         }

         k_h_t = k_h; 
         if((k == Kb - 1) && k_pad) {
            k_h_t = k_h1;
         }

         n_h_t = n_h;
         if((n == Nb - 1) && n_pad) {
            n_h_t = n_h1;
         }


         if(k == 0) {
            if(buf_C) {
               C_h1 = (float*) calloc(m_h_t * n_h_t , sizeof( float ));
               C_h = C_h1;
            } else {
               C_h2 = (float*) calloc(m_h_t * n_h_t , sizeof( float ));     
               C_h = C_h2;                               
            }
         }

         // mpi scatterv recv A_h
         MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, A_h, m_h_t * k_h_t, 
                     MPI_FLOAT, 0, comm_used);

         // pack A and reuse this packed copy for all B tiles in the CB block
         init_block_dims(m_h_t, n_h_t, k_h_t, p_dev, xa, cake_cntx, KMN);
         pack_A_single_buf_k_first(A_h, A_p, m_h_t, k_h_t, p_dev, xa, cake_cntx);

         
         int z1 = (int) (alpha_n*m_h);
         int num_B = (n_h_t / z1) + ((n_h_t % z1) ? 1 : 0);
         int n_rem = n_h_t % z1;
         int C_offset = 0;

         bool buf = 0; // controls buf choice
         B_h = B_h1;


         while(nb_ind < num_B) {

            int n_hx = z1;
            if((nb_ind == num_B - 1) && n_rem) {
               n_hx = n_rem;
            }

            // mpi bcast recv B_h
            MPI_Bcast(B_h, k_h_t*n_hx, MPI_FLOAT, 0, comm_used);

            if(nb_ind > 0) {
               pthread_join(gemm_thread, NULL);
            }

            inp->M = m_h_t;
            inp->N = n_hx;
            inp->K = k_h_t;
            inp->p = p_dev;
            inp->A = A_p;
            inp->B = B_h;
            inp->C = &C_h[C_offset];
            inp->cake_cntx = cake_cntx; 

            // switch buffers for double buffering
            B_h = buf ? B_h1 : B_h2;
            buf = !buf;

            // launch matmul on separate thread to overlap compute with IO
            if(pthread_create(&gemm_thread, NULL, cake_sgemm_launch, (void*) inp) != 0) {
               printf("Thread creation failed\n");
            }

            C_offset += m_h_t*n_hx;
            nb_ind++;
         }

         pthread_join(gemm_thread, NULL);

         k++;
         nb_ind = 0;

         if(k == Kb) {

            if(started) {
               MPI_Wait(&req, &status);
               if(buf_C) {
                  free(C_h2); 
               } else {
                  free(C_h1);
               }
            } else {
               started = 1;
            }

            // non-blocking gatherv C_h
            MPI_Igatherv(C_h, m_h_t * n_h_t, MPI_FLOAT,
                        NULL, NULL, NULL, MPI_FLOAT, 0, comm_used, &req);

            k = 0;
            m++;
            buf_C = !buf_C;
         }

         if(m == Mb) {
            m = 0;
            n++;
         }
      }
   }

   MPI_Wait(&req, &status);

   free(xa);
   free(cake_cntx);
   free(inp);
   free(A_h);
   free(A_p);
   free(B_h1);
   free(B_h2);



   if(taskid <= p_l) {
      MPI_Comm_free(&comm_pad);
   }
}
