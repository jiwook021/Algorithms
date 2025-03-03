#include <stdio.h>

// CUDA 커널: GPU에서 벡터 덧셈 수행
__global__ void VectorAdd(int *VectorA, int *VectorB, int *Result, int length) {
    int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (GlobalIndex < length) {
        Result[GlobalIndex] = VectorA[GlobalIndex] + VectorB[GlobalIndex];
    }
}

int main() {
    int n = 1000;
    int *a, *b, *c;
    int size = n * sizeof(int);

    // 호스트 메모리 할당
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    // 배열 초기화
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // 디바이스 메모리 할당
    int *DA, *DB, *DC;
    cudaMalloc(&DA, size);
    cudaMalloc(&DB, size);
    cudaMalloc(&DC, size);

    // 데이터 복사 (호스트 -> 디바이스)
    cudaMemcpy(DA, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(DB, b, size, cudaMemcpyHostToDevice);

    // 커널 실행
    VectorAdd<<<1, n>>>(DA, DB, DC, n);

    // 결과 복사 (디바이스 -> 호스트)
    cudaMemcpy(c, DC, size, cudaMemcpyDeviceToHost);

    // 결과 출력
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // 메모리 해제
    free(a);
    free(b);
    free(c);
    cudaFree(DA);
    cudaFree(DB);
    cudaFree(DC);

    return 0;
}