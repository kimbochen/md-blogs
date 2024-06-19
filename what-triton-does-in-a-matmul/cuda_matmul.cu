#include <stdio.h>
#include <stdlib.h>

#define CDIV(a, b) ( ((a) + (b) - 1) / (b) )
#define TILE_SIZE 32


__global__ void tiledSgemmKernel(const float *A, const float *B, float *C, int32_t N)
{
    __shared__ float A_tile[TILE_SIZE * TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE * TILE_SIZE];
    float accum = 0.0;

    int32_t i = blockDim.y * blockIdx.y + threadIdx.y;
    int32_t j = blockDim.x * blockIdx.x + threadIdx.x;

    int32_t tile_i = threadIdx.y;
    int32_t tile_j = threadIdx.x;
    int32_t tile_size = blockDim.x;

    for (int32_t k = 0; k < N; k += tile_size) {
        __syncthreads();
        A_tile[tile_i * tile_size + tile_j] = A[i * N + (k + tile_j)];
        B_tile[tile_i * tile_size + tile_j] = B[(k + tile_i) * N + j];
        __syncthreads();

        for (int32_t tile_k = 0; tile_k < tile_size; tile_k++) {
            accum += A_tile[tile_i * tile_size + tile_k] * B_tile[tile_k * tile_size + tile_j];
        }
    }

    C[i * N + j] = accum;
}

void tiledSgemm(const float *A, const float *B, float *C, int32_t N)
{
    int32_t BLOCK_SIZE = TILE_SIZE;

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(CDIV(N, BLOCK_SIZE), CDIV(N, BLOCK_SIZE));

    tiledSgemmKernel<<<grid_dim, block_dim>>>(A, B, C, N);
}


int main(int argc, char **argv)
{
    int32_t N = 4096;
    int32_t nelems = N * N;

    float *A_h = (float*) malloc(nelems * sizeof(float));
    float *B_h = (float*) malloc(nelems * sizeof(float));
    float *C_h = (float*) malloc(nelems * sizeof(float));

    for (int i = 0; i < nelems; i++) {
        A_h[i] = (rand() % 100) / 100.00;
        B_h[i] = (rand() % 100) / 100.00;
    }

    float *A_d, *B_d, *C_d;

    cudaMalloc((void**)&A_d, nelems * sizeof(float));
    cudaMalloc((void**)&B_d, nelems * sizeof(float));
    cudaMalloc((void**)&C_d, nelems * sizeof(float));

    cudaMemcpy(A_d, A_h, nelems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, nelems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, nelems * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    tiledSgemm(A_d, B_d, C_d, N);

    cudaDeviceSynchronize();

    cudaMemcpy(A_h, A_d, nelems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_h, B_d, nelems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_h, C_d, nelems * sizeof(float), cudaMemcpyDeviceToHost);


    float *C_v = (float*) malloc(nelems * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_v[i * N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                C_v[i * N + j] += A_h[i * N + k] * B_h[k * N + j];
            }
        }
    }

    // puts("C_h:");
    // for (int i = 0; i < nelems; i++) {
    //     printf("%.3f\t", C_h[i]);
    //     if ((i + 1) % N == 0) {
    //         puts("");
    //     }
    // }
    // puts("");

    // puts("C_v:");
    // for (int i = 0; i < nelems; i++) {
    //     printf("%.3f\t", C_v[i]);
    //     if ((i + 1) % N == 0) {
    //         puts("");
    //     }
    // }

    for (int i = 0; i < nelems; i++) {
        float diff = C_v[i] - C_h[i];
        if (diff > 1e-3 || diff < -1e-3) {
            puts("Wrong answer.");
            break;
        }
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A_h);
    free(B_h);
    free(C_h);
    free(C_v);

    return 0;
}
