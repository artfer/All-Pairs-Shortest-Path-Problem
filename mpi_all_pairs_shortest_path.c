#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void main(int argc, char **argv) {
  int numprocs, rank, tag, N;
  double start, end;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  tag = rank;

  if(rank==0){
    scanf("%d",&N);
    int m[N][N];

    for(int i=0; i<N; i++){
      for(int j=0; j<N; j++){
        scanf("%d",&m[i][j]);
      }
    }

  }



  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
  return;
}
