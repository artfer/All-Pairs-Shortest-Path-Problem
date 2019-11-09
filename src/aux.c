#include <math.h>


#define MAX 65536
#define MIN(x, y) ((x) < (y)) ? (x) : (y)


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


int Get_value(int* mat, int i, int j, int n){
    int offset = (i * n) + j;
    return mat[offset];
}


void Put_value(int* mat, int i, int j, int n, int val){
    int offset = (i * n) + j;
    mat[offset] = val;
}


void Update_value(int* mat, int i, int j, int n, int val){
    int offset = (i * n) + j;
    mat[offset] = MIN(mat[offset], val);
}


void Set_to_max(int* mat, int n){
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            Put_value(mat,i,j,n,MAX);
}


void Local_matrix_multiply(int* local_A, int* local_B, int* local_C, int n){
    int tmp; 

    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++){
                
                tmp = Get_value(local_A, i, k, n) + 
                    Get_value(local_B, k, j, n);

                Update_value(local_C, i, j, n, tmp);     
            }
}