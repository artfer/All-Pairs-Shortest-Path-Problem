#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "aux.c"


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


void Read_matrix(int* matrix, GRID_INFO_TYPE* grid, int n);
void Print_matrix(int* matrix, GRID_INFO_TYPE* grid, int n);
void Setup_grid(GRID_INFO_TYPE* grid, int n); 
void Fox(int n, GRID_INFO_TYPE* grid, int* local_A, int* local_B, int* local_C);
void Scatter_matrix(int* matrix, int* local_A, int* local_B, GRID_INFO_TYPE grid, int n);
void Gather_matrix(int* matrix, int* local_A, GRID_INFO_TYPE grid, int n);
void Min_plus_matrix_mul(int* matrix, int n, GRID_INFO_TYPE grid);


void Read_matrix(int* matrix, GRID_INFO_TYPE* grid, int n){
    int tmp;

    if(grid->my_rank == 0)
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++){
                scanf("%d", &tmp);

                tmp = Transform(i, j, tmp);                                 
                Put_value(matrix, i, j, n, tmp);
            }
}


void Print_matrix(int* matrix, GRID_INFO_TYPE* grid, int n){
    int tmp;

    if(grid->my_rank == 0) 
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                tmp = Get_value(matrix, i, j, n);
                tmp = Transform_inverse(tmp);
                printf("%d ", tmp);
            }
            printf("\n");
        }
}


void Setup_grid(GRID_INFO_TYPE* grid, int n){
    int old_rank;
    int dimensions[2];
    int periods[2];
    int coordinates[2];
    int varying_coords[2];

    // Set up Global Grid Information
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->p)); 
    MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);

    // Validate
    if(  n % (int) sqrt(grid->p) != 0){
        if(old_rank == 0){
            printf("ERROR: Invalid configuration!\n");
        }
        MPI_Finalize();
        exit(0);
    }
    grid->q = (int) sqrt((double) grid->p);
    dimensions[0] = dimensions[1] = grid->q;
    
    periods[0] = periods[1] = 1;

    // Create a cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 0, &(grid->comm));

    // Determines the rank of the calling process
    MPI_Comm_rank(grid->comm, &(grid->my_rank));

    // Determines process coords in cartesian topology given rank in group
    MPI_Cart_coords(grid->comm, grid->my_rank, 2, coordinates);
    grid->my_row = coordinates[0];
    grid->my_col = coordinates[1];

    // Set up row and column communicators
    varying_coords[0] = 0;
    varying_coords[1] = 1;
    MPI_Cart_sub(grid->comm, varying_coords, &(grid->row_comm));
    varying_coords[0] = 1;
    varying_coords[1] = 0;
    MPI_Cart_sub(grid->comm, varying_coords, &(grid->col_comm));

    MPI_Barrier(MPI_COMM_WORLD);
}


void Fox(int n, GRID_INFO_TYPE* grid, int* local_A, int* local_B, int* local_C){
  
    int step; 
    int bcast_root;
    int n_bar; // order of block submatrix = n/q
    int source,dest;
    MPI_Status status;

    n_bar = n/grid->q;
    Set_to_max(local_C, n_bar);

    

    // Calculate addresses for circular shift of B
    source = (grid->my_row + 1) % grid->q;
    dest = (grid->my_row + grid->q - 1) % grid->q;

    // Set aside storage for the broadcast block of A 
    int* temp_A = (int*) malloc(n_bar * n_bar * sizeof(int));

    for(step = 0; step < grid->q; step++){

        bcast_root = (grid->my_row + step) % grid->q;
        if(bcast_root == grid->my_col) {
            MPI_Bcast(local_A, n_bar * n_bar, MPI_INT,bcast_root, grid->row_comm);
            Local_matrix_multiply(local_A,local_B,local_C, n_bar);
        } else {
            MPI_Bcast(temp_A, n_bar * n_bar, MPI_INT, bcast_root, grid->row_comm);
            Local_matrix_multiply(temp_A,local_B,local_C, n_bar);
        }
        MPI_Sendrecv_replace(local_B, n_bar * n_bar, MPI_INT,
                dest, 0, source, MPI_ANY_TAG, grid->col_comm, &status);
        
    } 
    free(temp_A);
}


void Scatter_matrix(int* matrix, int* local_A, int* local_B, GRID_INFO_TYPE grid, int n){
    int n_bar = n / grid.q;
    int dest_coords[2];
    int tmp;
    int dest;
    MPI_Status status;

    if(grid.my_rank == 0){

        for(int i = 0; i < n; i++){
            dest_coords[0] = i / n_bar;
            
            for(int j = 0; j < n; j++){
                dest_coords[1] = j / n_bar;

                // get rank of destination process
                MPI_Cart_rank(grid.comm, dest_coords, &dest);
                    
                tmp = Get_value(matrix, i, j, n);

                if(dest == 0){
                    Put_value(local_A, i%n_bar, j%n_bar, n_bar, tmp);
                    Put_value(local_B, i%n_bar, j%n_bar, n_bar, (int) tmp);
                } else
                    MPI_Send(&tmp, 1, MPI_INT, dest, 0, grid.comm);
            }
        }
    }
    else 
        for(int i = 0; i < n_bar; i++)
            for(int j = 0; j < n_bar; j++){
                MPI_Recv(&tmp, 1, MPI_INT, 0, 0, grid.comm, &status);
                Put_value(local_A, i, j, n_bar, tmp);
                Put_value(local_B, i, j, n_bar, (int) tmp);
            }
}


void Gather_matrix(int* matrix, int* local_C, GRID_INFO_TYPE grid, int n){
    int n_bar = n / grid.q;
    int src_coords[2];
    int tmp;
    int src;
    MPI_Status status;

    if(grid.my_rank == 0){

        for(int i = 0; i < n; i++){
            src_coords[0] = i / n_bar;
            
            for(int j = 0; j < n; j++){
                src_coords[1] = j / n_bar;
                
                // get rank of source process
                MPI_Cart_rank(grid.comm, src_coords, &src);
                
                if(src == 0)
                    tmp = Get_value(local_C, i%n_bar, j%n_bar, n_bar);
                else
                    MPI_Recv(&tmp, 1, MPI_INT, src, 0, grid.comm, &status);
                Put_value(matrix, i, j, n, tmp);
            }
        }
    } else 
        for(int i = 0; i < n_bar; i++)
            for(int j = 0; j < n_bar; j++){
                tmp = Get_value(local_C, i, j, n_bar);
                MPI_Send(&tmp, 1, MPI_INT, 0, 0, grid.comm);
            }
}

void Min_plus_matrix_mul(int* matrix, int n, GRID_INFO_TYPE grid){
    int n_bar = n / grid.q;

    // Allocate space for local matrices
    int* local_A = (int *)malloc(n_bar * n_bar * sizeof(int));
    int* local_B = (int *)malloc(n_bar * n_bar * sizeof(int));
    int* local_C = (int *)malloc(n_bar * n_bar * sizeof(int));

    for(int f = 2; f < n; f+=f){

        // Sequential
        //Local_matrix_multiply(matrix, matrix, matrix, n);

        // MPI
        Scatter_matrix(matrix, local_A, local_B, grid, n);

        Fox(n, &grid, local_A, local_B, local_C);
  
        Gather_matrix(matrix, local_C, grid, n);
        
    } 
    free(local_A);
    free(local_B);
    free(local_C);
}


void main(int argc, char **argv) {
    int rank, n;
    GRID_INFO_TYPE grid;
    double start, end;
    int* matrix;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0){
        scanf("%d", &n);
        matrix = (int *) malloc(n * n * sizeof(int));
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Setup_grid(&grid, n);

    Read_matrix(matrix, &grid, n);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    Min_plus_matrix_mul(matrix, n, grid);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    double total = end - start;
    if(grid.my_rank == 0)
    //    printf("Time %.7f\n",total);

    Print_matrix(matrix, &grid, n);

    free(matrix);

    MPI_Finalize();
    return;
}