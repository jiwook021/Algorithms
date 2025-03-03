/**
 * @file main.cu
 * @brief Driver for CUDA vector addition.
 */

#include "VectorAdd.cuh"
#include <cstdio>
#include <cstdlib>

int main() {
    const int N = 1000;

    int* a = (int*)malloc(N * sizeof(int));
    int* b = (int*)malloc(N * sizeof(int));
    int* c = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    VectorAdd(a, b, c, N);

    // Print first 10 results
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Verify
    bool pass = true;
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) {
            fprintf(stderr, "Mismatch at index %d: %d != %d\n", i, c[i], a[i] + b[i]);
            pass = false;
            break;
        }
    }
    printf("Verification: %s\n", pass ? "PASSED" : "FAILED");

    free(a);
    free(b);
    free(c);
    return pass ? 0 : 1;
}
