#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int p;              // Total number of processes
  MPI_Comm comm;      // Comunicator for entire grid
  MPI_Comm row_comm;  // Comunicator for my row
  MPI_Comm col_comm;  // Comunicator for my col
  int q;              // Order of grid
  int my_row;         // My row number
  int my_col;         // My column number
  int my_rank;        // My rank in the grid communicator
} GRID_INFO_TYPE;

typedef struct {
  
} LOCAL_MATRIX_TYPE;

typedef struct {
  
} DERIVED_LOCAL_MATRIX;


void Setup_grid(GRID_INFO_TYPE* grid);
void Fox(int n, GRID_INFO_TYPE* grid,
    LOCAL_MATRIX_TYPE* local_A,
    LOCAL_MATRIX_TYPE* local_B,
    LOCAL_MATRIX_TYPE* local_C);
void Local_matrix_multiply(LOCAL_MATRIX_TYPE* local_A,
    LOCAL_MATRIX_TYPE* local_B,
    LOCAL_MATRIX_TYPE* local_C);
LOCAL_MATRIX_TYPE* Local_matrix_allocate(int n_bar);
void Set_to_zero(LOCAL_MATRIX_TYPE* local_A);

void Setup_grid(GRID_INFO_TYPE* grid){
  int old_rank;
  int dimensions[2];
  int periods[2];
  int coordinates[2];
  int varying_coords[2];

  // Set up Global Grid Information
  MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));
  MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);

  // Assuming it's a perfect square...
  grid->q = (int) sqrt((double) grid->p);
  dimensions[0] = dimensions[1] = grid->p;

  periods[0] = periods[1] = 1;

  // create a cartesian communicator
  MPI_Cart_create(MPI_COMM_WORLD,2,dimensions,periods,1,&(grid->comm));

  // Determines the rank of the calling process
  MPI_Comm_rank(grid->comm, &(grid->my_rank));

  // Determines process coords in cartesian topology given rank in group
  MPI_Cart_coords(grid->comm,grid->my_rank,2,coordinates);
  grid->my_row = coordinates[0];
  grid->my_col = coordinates[1];

  // Set up row and column communicators
  varying_coords[0] = 0;
  varying_coords[1] = 1;
  MPI_Cart_sub(grid->comm,varying_coords,&(grid->row_comm));
  varying_coords[0] = 1;
  varying_coords[1] = 0;
  MPI_Cart_sub(grid->comm,varying_coords,&(grid->col_comm));
}


void Fox(int n, GRID_INFO_TYPE* grid,
    LOCAL_MATRIX_TYPE* local_A,
    LOCAL_MATRIX_TYPE* local_B,
    LOCAL_MATRIX_TYPE* local_C){
  
  LOCAL_MATRIX_TYPE* temp_A;
  int step; 
  int bcast_root;
  int n_bar; // order of block submatrix = n/q
  int source;
  int dest;
  int tag = 43;
  MPI_Status status;

  n_bar = n/grid->q;
  Set_to_zero(local_C);

  // Calculate addresses for circular shift of B
  source = (grid->my_row + 1) % grid->q;
  dest = (grid->my_row + grid->q - 1) % grid->q;

  // Set aside storage for the broadcast block of A 
  temp_A = Local_matrix_allocate(n_bar);

  for(step = 0; step < grid->q; step++){
    bcast_root = (grid->my_row + step) % grid->q;
    if(bcast_root == grid->my_col) {
      MPI_Bcast(local_A, 1, DERIVED_LOCAL_MATRIX,bcast_root, grid->row_comm);
      Local_matrix_multiply(local_A,local_B,local_C);
    } else {
      MPI_Bcast(temp_A, 1, DERIVED_LOCAL_MATRIX, bcast_root, grid->row_comm);
      Local_matrix_multiply(temp_A,local_B,local_C);
    }
    MPI_Send(local_B, 1, DERIVED_LOCAL_MATRIX, dest, tag, grid->col_comm);
    MPI_Recv(local_B, 1, DERIVED_LOCAL_MATRIX, source, tag, grid->col_comm, &status);
  } 
}
    


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
