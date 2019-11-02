#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX 65536
#define MIN(x, y) ((x) < (y)) ? (x) : (y)

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
  int n_bar;
  #define Order(A) ((A)->n_bar)
  float entries[MAX];
  #define Entry(A,i,j) (*(((A)->entries) + ((A)->n_bar)*(i) + (j))) 
} LOCAL_MATRIX_TYPE;


LOCAL_MATRIX_TYPE* Local_matrix_allocate(int n_bar);
void Set_to_zero(LOCAL_MATRIX_TYPE* local_A);
void Setup_grid(GRID_INFO_TYPE* grid);
void Fox(int n, GRID_INFO_TYPE* grid,
    LOCAL_MATRIX_TYPE* local_A,
    LOCAL_MATRIX_TYPE* local_B,
    LOCAL_MATRIX_TYPE* local_C);
void Local_matrix_multiply(LOCAL_MATRIX_TYPE* local_A,
    LOCAL_MATRIX_TYPE* local_B,
    LOCAL_MATRIX_TYPE* local_C);

MPI_Datatype DERIVED_LOCAL_MATRIX;


LOCAL_MATRIX_TYPE* Local_matrix_allocate(int n_bar){
  LOCAL_MATRIX_TYPE * tmp;
  
  tmp = (LOCAL_MATRIX_TYPE*) malloc(sizeof(LOCAL_MATRIX_TYPE));
  return tmp;
}


void Set_to_zero(LOCAL_MATRIX_TYPE*  local_A) {
    int i, j;

    for (i = 0; i < Order(local_A); i++)
        for (j = 0; j < Order(local_A); j++)
            Entry(local_A,i,j) = 0.0;
}


void Local_matrix_multiply(
         LOCAL_MATRIX_TYPE*  local_A,
         LOCAL_MATRIX_TYPE*  local_B, 
         LOCAL_MATRIX_TYPE*  local_C) {
    int i, j, k;

    for (i = 0; i < Order(local_A); i++)
        for (j = 0; j < Order(local_A); j++)
            for (k = 0; k < Order(local_B); k++){
                int tmp = Entry(local_A,i,k)+Entry(local_B,k,j);
                Entry(local_C,i,j) = MIN(Entry(local_C,i,j), tmp);
            }

}


void Read_matrix(LOCAL_MATRIX_TYPE*  local_A, GRID_INFO_TYPE* grid, int n) {

    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        dest;
    int        coords[2];
    float*     temp;
    MPI_Status status;
    
    if (grid->my_rank == 0) {
        temp = (float*) malloc(Order(local_A)*sizeof(float));
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/Order(local_A);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &dest);
                if (dest == 0) {
                    for (mat_col = 0; mat_col < Order(local_A); mat_col++)
                        scanf("%f", 
                          (local_A->entries)+mat_row*Order(local_A)+mat_col);
                } else {
                    for(mat_col = 0; mat_col < Order(local_A); mat_col++)
                        scanf("%f", temp + mat_col);
                    MPI_Send(temp, Order(local_A), MPI_FLOAT, dest, 0,
                        grid->comm);
                }
            }
        }
        free(temp);
    } else {
        for (mat_row = 0; mat_row < Order(local_A); mat_row++) 
            MPI_Recv(&Entry(local_A, mat_row, 0), Order(local_A), 
                MPI_FLOAT, 0, 0, grid->comm, &status);
    }          
} 


void Print_matrix(
         LOCAL_MATRIX_TYPE* local_A, GRID_INFO_TYPE* grid, int n) {
    int mat_row, mat_col;
    int grid_row, grid_col;
    int source;
    int coords[2];
    float* temp;
    MPI_Status status;

    if (grid->my_rank == 0) {
        temp = (float*) malloc(Order(local_A)*sizeof(float));
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/Order(local_A);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &source);
                if (source == 0) {
                    for(mat_col = 0; mat_col < Order(local_A); mat_col++)
                        printf("%4.1f ", Entry(local_A, mat_row, mat_col));
                } else {
                    MPI_Recv(temp, Order(local_A), MPI_FLOAT, source, 0,
                        grid->comm, &status);
                    for(mat_col = 0; mat_col < Order(local_A); mat_col++)
                        printf("%4.1f ", temp[mat_col]);
                }
            }
            printf("\n");
        }
        free(temp);
    } else {
        for (mat_row = 0; mat_row < Order(local_A); mat_row++) 
            MPI_Send(&Entry(local_A, mat_row, 0), Order(local_A), 
                MPI_FLOAT, 0, 0, grid->comm);
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
  int p,rank,n,n_bar;
  GRID_INFO_TYPE grid;
  LOCAL_MATRIX_TYPE* local_A;
  LOCAL_MATRIX_TYPE* local_B;
  LOCAL_MATRIX_TYPE* local_C;
  double start, end;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Setup_grid(&grid);
  if (rank == 0)
    scanf("%d", &n);

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  n_bar = n/grid.q;

  local_A = Local_matrix_allocate(n_bar);
  Order(local_A) = n_bar;
  Read_matrix(local_A, &grid, n);

  Build_matrix_type(local_A);
  //tmp = Local_matrix_allocate(n_bar);

  local_C = Local_matrix_allocate(n_bar);
  Order(local_C) = n_bar;
  Fox(n, &grid, local_A, local_A, local_C);

  Print_matrix(local_C, &grid, n);

  Free_local_matrix(&local_A);
  Free_local_matrix(&local_C);

  MPI_Finalize();
  return;
}