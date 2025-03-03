#include <stdio.h>

// CUDA 커널: GPU에서 벡터 덧셈 수행
__global__ void vectorAdd(int *vectorA, int *vectorB, int *result, int length) {
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIndex < length) {
        result[globalIndex] = vectorA[globalIndex] + vectorB[globalIndex];
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
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 데이터 복사 (호스트 -> 디바이스)
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 커널 실행
    vectorAdd<<<1, n>>>(d_a, d_b, d_c, n);

    // 결과 복사 (디바이스 -> 호스트)
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // 결과 출력
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // 메모리 해제
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}