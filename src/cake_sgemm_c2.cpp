#include "cake_c2.h"
#include "cake.h"

int main(int argc, char *argv[]) {

   // int numprocs,              /* number of tasks in partition */
   // rank;                /* a task identifier */
   
   // MPI_Init(&argc,&argv);
   // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   // MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

   // int p_l = 2;
   // int color = rank <= p_l ? 1 : MPI_UNDEFINED; // Determine color based on row

   // // Split the communicator based on the color and use the
   // // original rank for ordering
   // MPI_Comm comm_pad;
   // MPI_Comm_split(MPI_COMM_WORLD, color, rank, &comm_pad);


   // if(rank <= p_l) {
   //    int pad_rank, p;
   //    MPI_Comm_rank(comm_pad, &pad_rank);
   //    MPI_Comm_size(comm_pad, &p);
   //    printf("proc : %d \t pad rank: %d with p_l = %d\n", rank, pad_rank, p);
   //    MPI_Comm_free(&comm_pad);
   // }

   // printf("proc : %d \n", rank);


   // exit(1);

   int numtasks,              /* number of tasks in partition */
   taskid,                /* a task identifier */
   rc;                 /* err */


   // MPI setup
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
   MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

   if(numtasks < 2 ) {
      printf("Need at least two MPI tasks. Quitting...\n");
      MPI_Abort(MPI_COMM_WORLD, rc);
      exit(1);
   }

   int M, K, N, p;
   int mat_inputs[4];

   if(taskid == 0) {
      
      if(argc < 4) {
         printf("Enter M, K, N and number of hosts\n");
         exit(1);
      }

      M = atoi(argv[1]);
      K = atoi(argv[2]);
      N = atoi(argv[3]);
      p = atoi(argv[4]); // number of devices on the network

      mat_inputs[0] = M;
      mat_inputs[1] = K;
      mat_inputs[2] = N;
      mat_inputs[3] = p;
   }

   // bcast M,N,K,p
   MPI_Bcast(mat_inputs, 4, MPI_INT, 0, MPI_COMM_WORLD);
   
   // hosts bcast recv M,N,K,p from server
   if(taskid > 0) {
      M = mat_inputs[0];
      K = mat_inputs[1];
      N = mat_inputs[2];
      p = mat_inputs[3];
   }

   int m_h, k_h, n_h;
   double alpha_n = 1.0;
   m_h = get_block_dim_C2(1ULL << 31, alpha_n, p);
   k_h = m_h;
   n_h = (int) (alpha_n * p * m_h);

   int p_dev = 2; // number of cores on a single device
   int mr_dev = 6; //4; // mr on raspbi device
   int m_r = mr_dev*p_dev > m_h ? m_h : mr_dev*p_dev;

   int k_pad = (K % k_h) ? 1 : 0; 
   int n_pad = (N % n_h) ? 1 : 0; 
   int m_pad = (M % (p*m_h)) ? 1 : 0; 

   int Mb = (M / (p*m_h)) + m_pad;
   int Nb = (N / n_h) + n_pad;
   int Kb = (K / k_h) + k_pad;

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


   // creates an extra communicator for p_l hosts involved in computing the m-padded region
   int color = taskid <= p_l ? 1 : MPI_UNDEFINED; // Determine color based on rank and p_l 
   MPI_Comm comm_pad, comm_used;
   MPI_Comm_split(MPI_COMM_WORLD, color, taskid, &comm_pad);



   if(taskid == 0) {
      
      printf("M = %d, K = %d, N = %d\n", M,K,N);
      printf("mh = %d\n", m_h);

      omp_set_num_threads(10); // TODO: use 10 threads only on intel i9 10900K

      struct timespec start, end;
      double diff_t;


      float* A = (float*) malloc(M * K * sizeof( float ));
      float* B = (float*) malloc(K * N * sizeof( float ));
      float* C = (float*) calloc(M * N , sizeof( float ));

      // initialize A and B
      srand(time(NULL));
      rand_init(A, M, K);
      rand_init(B, K, N);

      // copy A,B,C for packing
      float* A_p = (float*) malloc(M * K * sizeof( float ));
      float* B_p = (float*) malloc(K * N * sizeof( float ));
      float* C_p = (float*) calloc(M * N , sizeof( float ));

      clock_gettime(CLOCK_REALTIME, &start);

      // pack A and B before sending to hosts
      pack_A_h(A, A_p, M, K, m_h, k_h, m_r, p);
      pack_B_h(B, B_p, K, N, m_h, k_h, n_h, alpha_n, p);

      // printf("HEYY\n");
      // for(int i = 0; i < K; i++) {
      //    for(int j = 0; j < N; j++) {
      //       printf("%.2f ", B[i*N + j]);
      //    }
      //    printf("\n");
      // }

      // printf("\n\nNOOO\n");
      // for(int i = 0; i < K; i++) {
      //    for(int j = 0; j < N; j++) {
      //       printf("%.2f ", B_p[i*N + j]);
      //    }
      //    printf("\n");
      // }

      // printf("HEYY\n");
      // for(int i = 0; i < M; i++) {
      //    for(int j = 0; j < K; j++) {
      //       printf("%f ", A[i*K + j]);
      //    }
      //    printf("\n");
      // }

      // printf("\n\nNOOO\n");
      // for(int i = 0; i < M; i++) {
      //    for(int j = 0; j < K; j++) {
      //       printf("%f ", A_p[i*K + j]);
      //    }
      //    printf("\n");
      // }

      if(DEBUG) printf("m_h = %d, k_h = %d, n_h = %d\n", m_h, k_h, n_h);

      // exit(1);



      int A_offset = 0;
      int B_offset = 0;
      int C_offset = 0;
      int* sendcounts = (int*) calloc((p + 1) , sizeof(int));
      int* displs = (int*) calloc((p + 1) , sizeof(int));
      int* recvcounts = (int*) calloc((p + 1) , sizeof(int));

      int m, k, n;
      int m_cb, n_h_t, k_h_t, p_used, host;

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
            } else {
               p_used = p;
               m_cb = p_used*m_h;
               comm_used = MPI_COMM_WORLD;
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

                  displs[host+1] = curr_disp;
                  curr_disp += sendcounts[host+1];
               }

               MPI_Scatterv(&A_p[A_offset], sendcounts, displs,
                     MPI_FLOAT, NULL, 0, MPI_FLOAT, 0, comm_used);

               A_offset += m_cb*k_h_t;
               memset(sendcounts, 0, (p + 1) * sizeof(int));
               memset(displs, 0, (p + 1) * sizeof(int));


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

            // // gatherv C
            int curr_disp = 0;
            for(host = 0; host < p_used; host++) {

               if((m == Mb - 1) && m_pad) {
                  recvcounts[host+1] = (host == (p_l - 1) ? m_h1_last_host*n_h_t : m_h1*n_h_t);
               } else {
                  recvcounts[host+1] = m_h*n_h_t;
               }

               displs[host+1] = curr_disp;
               curr_disp += recvcounts[host+1];
            }

            // printf("recvcnts CC HEY \n");
            // printf("n_h_t = %d, m_h = %d, m_h1 = %d, m_h1_last_host = %d, p_l = %d\n\n", n_h_t, m_h, m_h1, m_h1_last_host, p_l);
            // for(int x = 0; x < (p + 1); x++) {
            //    printf("%d ", recvcounts[x]);
            // }
            // printf("\n\n\n\n");

            MPI_Gatherv(NULL, 0, MPI_FLOAT, &C_p[C_offset], recvcounts, displs,
               MPI_FLOAT, 0, comm_used);

            C_offset += m_cb*n_h_t;

            memset(recvcounts, 0, (p + 1)*sizeof(int));
            memset(displs, 0, (p + 1)*sizeof(int));
         }
      }

      // printf("\n\n\n\n");
      // float* C1 = (float*) calloc(M * N , sizeof( float ));
      // cake_cntx_t* cake_cntx = cake_query_cntx();
      // cake_sgemm(A, B, C1, M, N, K, 10, cake_cntx);
      // for(int i = 0; i < M*N; i++) {
      //    printf("%f ", C1[i]);
      // }
      // printf("\n\n\n\n");


      unpack_C_h(C, C_p, M, N, m_h, n_h, m_r, alpha_n, p);


      // printf("\n\n\n\n");

      // for(int i = 0; i < M*N; i++) {
      //    printf("%f ", C[i]);
      // }
      // printf("\n\n\n\n");

      clock_gettime(CLOCK_REALTIME, &end);
      long seconds = end.tv_sec - start.tv_sec;
      long nanoseconds = end.tv_nsec - start.tv_nsec;
      diff_t = seconds + nanoseconds*1e-9;
      printf("sgemm time: %f \n", diff_t); 


      cake_sgemm_checker(A, B, C, N, M, K);

      free(A_p);
      free(B_p);
      free(C_p);
      free(A);
      free(B);
      free(C);
   }



   if (taskid > 0) {

      int host = taskid - 1;

      float *A_h, *B_h, *C_h, *A_p;

      int m = 0, k = 0, n = 0;
      int nb_ind = 0;
      int n_h_t, k_h_t, m_h_t;


      // A_h = (float*) calloc(m_h * k_h , sizeof(float));
      // B_h = (float*) malloc(k_h*n_h * sizeof(float));
      // posix_memalign((void**) &A_p, 64, (m_h+mr_dev) * k_h * sizeof(float));

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
               C_h = (float*) calloc(m_h_t * n_h_t , sizeof( float ));
            }

            A_h = (float*) malloc(m_h_t * k_h_t * sizeof( float ));
            
            // mpi scatterv recv A_h
            MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, A_h, m_h_t * k_h_t, 
                        MPI_FLOAT, 0, comm_used);

            cake_cntx_t* cake_cntx = cake_query_cntx();
            blk_dims_t* blk_dims = get_block_dims(cake_cntx, m_h_t, p_dev);

            int A_sz = cake_sgemm_packed_A_size(m_h_t, k_h_t, p_dev, cake_cntx, blk_dims);
            posix_memalign((void**) &A_p, 64, A_sz);
            pack_A_single_buf(A_h, A_p, m_h_t, k_h_t, p_dev, cake_cntx, blk_dims);

            // if(n == 0 && m == 2 && host == 1) {
            //    printf("blah A scatter m = %d, n = %d from host = %d, m_h_t = %d, n_h_t = %d\n", m,n, host,m_h_t ,n_h_t);
            //    // exit(1);
            // }

            // if(host == 1) {
            //    printf("host %d, size = %d\n\n", host,m_h_t * k_h_t);
            //    for(int i = 0; i < m_h_t*k_h_t; i++) {
            //          printf("%f ", A_h[i]);  
            //    }
            //    printf("\n\n\n\n");
            // }

            // exit(1);

            int z1 = (int) (alpha_n*m_h);
            int num_B = (n_h_t / z1) + ((n_h_t % z1) ? 1 : 0);
            int n_rem = n_h_t % z1;
            int C_offset = 0;

            while(nb_ind < num_B) {

               int n_hx = z1;
               if((nb_ind == num_B - 1) && n_rem) {
                  n_hx = n_rem;
               }

               B_h = (float*) malloc(k_h_t*n_hx * sizeof(float));

               // mpi bcast recv B_h
               MPI_Bcast(B_h, k_h_t*n_hx, MPI_FLOAT, 0, comm_used);

               // execute matmul
               cake_sgemm(A_p, B_h, &C_h[C_offset], m_h_t, n_hx, k_h_t, p_dev, cake_cntx, true, false);
               // cake_sgemm(A_h, B_h, &C_h[C_offset], m_h_t, n_hx, k_h_t, p_dev, cake_cntx);
               // cake_sgemm_checker(A_h, B_h, &C_h[C_offset], n_hx, m_h_t, k_h_t);

               // if(host == 0) {
               //    if(cake_sgemm_checker(A_h, B_h, &C_h[C_offset], n_hx, m_h_t, k_h_t)) {
               //       printf("%d %d %d %d\n", n_hx, m_h_t, k_h_t, C_offset);

               //       for(int i = 0; i < k_h_t*m_h_t; i++) {
               //             printf("%.2f ", A_h[i]);  
               //       }
               //       printf("\n\n\n\n");


               //       for(int i = 0; i < k_h_t*n_hx; i++) {
               //             printf("%.2f ", B_h[i]);  
               //       }
               //       printf("\n\n\n\n");

               
               //       exit(1);
               //    };
               // }

               C_offset += m_h_t*n_hx;
               nb_ind++;
               // memset(B_h, 0, k_h * n_h * sizeof(float));
               free(B_h);
            }

            k++;
            nb_ind = 0;
            free(A_h);
            free(A_p);
            // memset(A_h, 0, k_h * m_h * sizeof(float));
            // memset(A_p, 0, k_h * (m_h+mr_dev) * sizeof(float));


            if(k == Kb) {

               // gatherv C_h
               MPI_Gatherv(C_h, m_h_t * n_h_t, MPI_FLOAT,
                           NULL, NULL, NULL, MPI_FLOAT, 0, comm_used);

               k = 0;
               m++;
               free(C_h);
            }

            if(m == Mb) {
               m = 0;
               n++;
            }
         }


      }
   }


   if(taskid <= p_l) {
      MPI_Comm_free(&comm_pad);
   }

   MPI_Finalize();

   return 0;
}

