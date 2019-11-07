#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


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


int  Transform(int i, int j, int val);
int  Transform_inverse(int val);
int  Get_value(int* local_A, int i, int j, int n_bar);
void Put_value(int* local_A, int i, int j, int n_bar, int val);
void Update_value(int* local_A, int i, int j, int n_bar, int val);
void Set_to_max(int* local_A, int n_bar);
void Local_matrix_multiply(int* local_A, int* local_B, int* local_C, int n_bar);
void Read_matrix(int* matrix, GRID_INFO_TYPE* grid, int n, int q);
void Print_matrix(int* matrix, GRID_INFO_TYPE* grid, int n, int q);
void Setup_grid(GRID_INFO_TYPE* grid, int n); 
void Fox(int n, GRID_INFO_TYPE* grid, int* local_A, int* local_B, int* local_C);
void Scatter_matrix(int* matrix, int* local_A, int* local_B, GRID_INFO_TYPE grid, int n);
void Gather_matrix(int* matrix, int* local_A, GRID_INFO_TYPE grid, int n);
void Min_plus_matrix_mul(int* matrix, int n, GRID_INFO_TYPE grid);


// in order to avoid a final matrix of 0's,
// every 0 (not in the main diagonal) turns to MAX 
int Transform(int i, int j, int val){
    if(i == j)
        return 0;
    if(val == 0)
        return MAX;
    return val;
}


int Transform_inverse(int val){
    return val == MAX ? 0 : val;
}


// same as returning local_A[i][j]
int Get_value(int* local_A, int i, int j, int n_bar){
    int offset = (i * n_bar) + j;
    return local_A[offset];
}


// inserts value in local_A[i][j]
void Put_value(int* local_A, int i, int j, int n_bar, int val){
    int offset = (i * n_bar) + j;
    local_A[offset] = val;
}


// updates local_A[i][j] to the min of its value and val
void Update_value(int* local_A, int i, int j, int n_bar, int val){
    int offset = (i * n_bar) + j;
    local_A[offset] = MIN(local_A[offset],val);
}


// initializes a matrix with MAX values
void Set_to_max(int* local_A, int n_bar){
    for(int i = 0; i < n_bar; i++)
        for(int j = 0; j < n_bar; j++)
            Put_value(local_A,i,j,n_bar,MAX);
}


// "multiply" local_A and local_B 
void Local_matrix_multiply(int* local_A, int* local_B, int* local_C, int n_bar){
    int tmp; 

    for(int i = 0; i < n_bar; i++)
        for(int j = 0; j < n_bar; j++)
            for(int k = 0; k < n_bar; k++){
                tmp = Get_value(local_A, i, k, n_bar) + 
                    Get_value(local_B, k, j, n_bar);
                Update_value(local_C, i, j, n_bar, tmp);     
            }
}


void Read_matrix(int* matrix, GRID_INFO_TYPE* grid, int n, int q){
    int n_bar = n / q;
    int tmp;

    if(grid->my_rank == 0){
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                scanf("%d",&tmp);

                tmp = Transform(i,j,tmp);                                 
                Put_value(matrix,i,j,n,tmp);
            }
        }
    }
}


void Print_matrix(int* matrix, GRID_INFO_TYPE* grid, int n, int q){
    int n_bar = n / q;
    int tmp;

    if(grid->my_rank == 0) {
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                tmp = Get_value(matrix,i,j,n);
                tmp = Transform_inverse(tmp);
                printf("%d ",tmp);
            }
            printf("\n");
        }
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

    // Assuming it's a perfect square...
    grid->q = (int) sqrt((double) grid->p);
    dimensions[0] = dimensions[1] = grid->q;
    
    periods[0] = periods[1] = 1;

    // create a cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD,2,dimensions,periods,0,&(grid->comm));

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

    MPI_Barrier(MPI_COMM_WORLD);
}


void Fox(int n, GRID_INFO_TYPE* grid, int* local_A, int* local_B, int* local_C){
  
    int step; 
    int bcast_root;
    int n_bar; // order of block submatrix = n/q
    int source,dest;
    int tag = 43;
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
        //MPI_Send(local_B, n_bar * n_bar, MPI_INT, dest, tag, grid->col_comm);
        //MPI_Recv(local_B, n_bar * n_bar, MPI_INT, source, tag, grid->col_comm, &status);
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
                //printf("%d   %d\n", dest_coords[0], dest_coords[1]);
                MPI_Cart_rank(grid.comm, dest_coords, &dest);
                    tmp = Get_value(matrix, i, j, n);
                    if(dest == 0){
                        Put_value(local_A, i % n_bar,  j % n_bar, n_bar, tmp);
                        Put_value(local_B, i % n_bar,  j % n_bar, n_bar, (int) tmp);
                    } else
                        MPI_Send(&tmp, 1, MPI_INT, dest, 0, grid.comm);
                    
                    
                    
                }
            }

    }
    else {
        for(int i = 0; i < n_bar; i++){
            for(int j = 0; j < n_bar; j++){
                MPI_Recv(&tmp, 1, MPI_INT, 0, 0, grid.comm, &status);
                //printf("tmp %d\n",tmp);
                Put_value(local_A, i, j, n_bar, tmp);
                Put_value(local_B, i, j, n_bar, (int) tmp);
                //printf("val %d\n",Get_value(local_A, i, j, n_bar));
            }
        }
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
                MPI_Cart_rank(grid.comm, src_coords, &src);
                if(src == 0){
                    
                    tmp = Get_value(local_C, i%n_bar, j%n_bar, n_bar);
                    //printf("i %d   j %d   tmp %d\n",i%n_bar,j%n_bar,tmp);
                }
                else
                    MPI_Recv(&tmp, 1, MPI_INT, src, 0, grid.comm, &status);
                //printf("x %d   y %d   tmp %d\n", src_coords[0], src_coords[1],tmp);
                Put_value(matrix, i, j, n, tmp);
            }
        }
    } else {
        for(int i = 0; i < n_bar; i++){
            for(int j = 0; j < n_bar; j++){
                tmp = Get_value(local_C, i, j, n_bar);
                MPI_Send(&tmp, 1, MPI_INT, 0, 0, grid.comm);
            }
        }
    }
}

void Min_plus_matrix_mul(int* matrix, int n, GRID_INFO_TYPE grid){
    int n_bar = n / grid.q;
    
    // Allocate space for local matrices
    int* local_A = (int *)malloc(n_bar * n_bar * sizeof(int));
    int* local_B = (int *)malloc(n_bar * n_bar * sizeof(int));
    int* local_C = (int *)malloc(n_bar * n_bar * sizeof(int));

    for(int f = 2; f < n; f+=f){

        //Print_matrix(matrix, &grid, n, grid.q);
        //printf("\n");
        // send submatrices to each process  
        //MPI_Scatter(matrix, n_bar*n_bar, MPI_INT,
        //            local_A, n_bar*n_bar, MPI_INT, 0, grid.comm);
        Scatter_matrix(matrix, local_A, local_B, grid, n);

        // calculate new matrix
        Fox(n, &grid, local_A, local_B, local_C);

        // get the final matrix
        //MPI_Gather(local_C, n_bar * n_bar, MPI_INT, 
        //            matrix, n_bar * n_bar, MPI_INT, 0, grid.comm);   
        Gather_matrix(matrix, local_C, grid, n);
    } 
    free(local_A);
    free(local_C);
}


int mat_cmp(int* matrix, int* matrix_out, int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if((int) Get_value(matrix,i,j,n) != (int) Get_value(matrix_out,i,j,n))
                return 0;
        }
    }
    return 1;
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
        matrix = (int *)malloc(n * n * sizeof(int));
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Setup_grid(&grid, n);

    Read_matrix(matrix, &grid, n, grid.q);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    Min_plus_matrix_mul(matrix, n, grid);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    double total = end - start;
    if(grid.my_rank == 0)
        printf("Time %.7f\n",total);

    // print the final matrix 
    Print_matrix(matrix, &grid, n, grid.q);

    /*if(rank == 0){
        scanf("%d", &n);
        int* out_matrix = (int *)malloc(n * n * sizeof(int));
        Read_matrix(out_matrix, &grid, n, grid.q);
        printf("%d\n",mat_cmp(matrix,out_matrix,n));
    }*/


    MPI_Finalize();
    return;
}