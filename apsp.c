#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX 65536
#define MIN(x, y) ((x) < (y)) ? (x) : (y)
#define ZERO_TO_MAX(x) (x == 0  ) ? MAX : x
#define MAX_TO_ZERO(x) (x == MAX) ?   0 : x 


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



void Read_matrix(int** a, GRID_INFO_TYPE* grid, int n) {

    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        dest;
    int        coords[2];
    int tmp;
    MPI_Status status;
    
    if (grid->my_rank == 0) {
        int ** tmp;
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/grid->q;
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &dest);
                if (dest == 0) {
                    for (mat_col = 0; mat_col < grid->q; mat_col++){
                        scanf("%d",tmp);
                        tmp = ZERO_TO_MAX(tmp);
                        a[mat_row][mat_col] = tmp;
                        //scanf("%f", (local_A->entries)+mat_row*Order(local_A)+mat_col);
                    }
                } else {
                    for(mat_col = 0; mat_col < grid->q; mat_col++){
                        scanf("%d",tmp);
                        tmp = ZERO_TO_MAX(tmp);
                        MPI_Send(tmp, 1, MPI_INT, dest, 0,grid->comm);
                    }
                }
            }
        }
    } else {  
        for (mat_row = 0; mat_row < grid->q; mat_row++) 
            for (mat_col = 0; mat_row < grid->q; mat_col++) 
                MPI_Recv(&a[mat_row][mat_col], 1, MPI_INT, 0, 0, grid->comm, &status);
    }          
} 

void Print_matrix(
         int** local_A, GRID_INFO_TYPE* grid, int n) {
    int mat_row, mat_col;
    int grid_row, grid_col;
    int source;
    int coords[2];
    int tmp;
    MPI_Status status;

    if (grid->my_rank == 0) {
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/grid->q;
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &source);
                if (source == 0) {
                    for(mat_col = 0; mat_col < grid->q; mat_col++)
                        printf("%d ", local_A[mat_row][mat_col]);
                } else {
                    
                    for(mat_col = 0; mat_col < grid->q; mat_col++)
                        MPI_Recv(tmp, 1, MPI_INT, source, 0, grid->comm, &status);
                        printf("%d ", tmp);
                }
            }
            printf("\n");
        }
    } else {
        for (mat_row = 0; mat_row < grid->q; mat_row++)
            for (mat_row = 0; mat_row < grid->q; mat_col++)  
            MPI_Send(local_A[mat_row][mat_col], 1, MPI_INT, 0, 0, grid->comm);
    }
                     
}


void Local_matrix_multiply(int** a, int** b, int** c,int q) {
    int i, j, k, x;

    for (i = 0; i < q; i++)
        for (j = 0; j < q; j++)
            for (k = 0; k < q; k++){
                x = a[i][k] + b[k][j];
                c[i][j] = MIN(c,x);
            }
}

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
    int** local_A,
    int** local_B,
    int** local_C){
  
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
  //temp_A = Local_matrix_allocate(n_bar);
  int** tmp;

  for(step = 0; step < grid->q; step++){
    bcast_root = (grid->my_row + step) % grid->q;
    if(bcast_root == grid->my_col) {
      MPI_Bcast(local_A, 1, MPI_INT,bcast_root, grid->row_comm);
      Local_matrix_multiply(local_A,local_B,local_C,grid->q);
    } else {
      MPI_Bcast(tmp, 1, MPI_INT, bcast_root, grid->row_comm);
      Local_matrix_multiply(tmp,local_B,local_C,grid->q);
    }
    MPI_Send(local_B, 1, MPI_INT, dest, tag, grid->col_comm);
    MPI_Recv(local_B, 1, MPI_INT, source, tag, grid->col_comm, &status);
  } 
}



void main(int argc, char **argv) {
  int p,rank,n,n_bar;
  GRID_INFO_TYPE grid;
  double start, end;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Setup_grid(&grid);
  if (rank == 0)
    scanf("%d", &n);

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  n_bar = n/grid.q;


  int local_A[grid.q][grid.q];
  int local_C[grid.q][grid.q];
  Read_matrix(local_A, &grid, n);
  //tmp = Local_matrix_allocate(n_bar);

  Fox(n, &grid, local_A, local_A, local_C);

  Print_matrix(local_C, &grid, n);

  MPI_Finalize();
  return;
}