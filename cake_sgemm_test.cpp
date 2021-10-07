#include "cake_c2.h"


int main(int argc, char *argv[]) {

   int numtasks,              /* number of tasks in partition */
   taskid,                /* a task identifier */
   rc = 0;                 /* err */

   // MPI setup
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
   MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

   if(numtasks < 2 ) {
      printf("Need at least two MPI tasks (1 server, 1 host). Quitting...\n");
      MPI_Abort(MPI_COMM_WORLD, rc);
      exit(1);
   }

   if(argc < 4) {
      printf("Enter M, K, N and number of hosts\n");
      exit(1);
   }

   int M, K, N, p;
   M = atoi(argv[1]);
   K = atoi(argv[2]);
   N = atoi(argv[3]);
   p = numtasks - 1; // number of devices on the network

   float *A = NULL, *B = NULL, *C = NULL;

   // root process on server initializes input matrices
   if(taskid == 0) {
      A = (float*) malloc(M * K * sizeof( float ));
      B = (float*) malloc(K * N * sizeof( float ));
      C = (float*) calloc(M * N , sizeof( float ));

      srand(time(NULL));
      rand_init(A, M, K);
      rand_init(B, K, N);      
   }

   cake_sgemm_net(A, B, C, M, N, K, p, taskid);

   MPI_Finalize();

   return 0;
}

