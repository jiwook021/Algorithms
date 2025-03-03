/**
 * @file test_VectorAdd.cu
 * @brief Unit tests for the VectorAdd CUDA kernel.
 */

#include "VectorAdd.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

static int testsPassed = 0;
static int testsFailed = 0;

#define ASSERT_EQ(actual, expected, msg)                                       \
    do {                                                                       \
        if ((actual) != (expected)) {                                          \
            fprintf(stderr, "[FAIL] %s: expected %d, got %d\n",               \
                    msg, (expected), (actual));                                \
            testsFailed++;                                                     \
        } else {                                                               \
            testsPassed++;                                                     \
        }                                                                      \
    } while (0)

void TestBasicAddition() {
    const int N = 100;
    int a[100], b[100], c[100];
    for (int i = 0; i < N; i++) { a[i] = i; b[i] = i * 2; }

    VectorAdd(a, b, c, N);

    for (int i = 0; i < N; i++) {
        ASSERT_EQ(c[i], i + i * 2, "BasicAddition");
    }
    printf("[PASSED] TestBasicAddition\n");
}

void TestZeroVector() {
    const int N = 50;
    int a[50], b[50], c[50];
    for (int i = 0; i < N; i++) { a[i] = i; b[i] = 0; }

    VectorAdd(a, b, c, N);

    for (int i = 0; i < N; i++) {
        ASSERT_EQ(c[i], i, "ZeroVector");
    }
    printf("[PASSED] TestZeroVector\n");
}

void TestNegativeValues() {
    const int N = 50;
    int a[50], b[50], c[50];
    for (int i = 0; i < N; i++) { a[i] = -i; b[i] = i; }

    VectorAdd(a, b, c, N);

    for (int i = 0; i < N; i++) {
        ASSERT_EQ(c[i], 0, "NegativeValues");
    }
    printf("[PASSED] TestNegativeValues\n");
}

void TestLargeVector() {
    const int N = 10000;
    int* a = (int*)malloc(N * sizeof(int));
    int* b = (int*)malloc(N * sizeof(int));
    int* c = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) { a[i] = 1; b[i] = 2; }

    VectorAdd(a, b, c, N);

    for (int i = 0; i < N; i++) {
        ASSERT_EQ(c[i], 3, "LargeVector");
    }
    printf("[PASSED] TestLargeVector\n");

    free(a); free(b); free(c);
}

int main() {
    TestBasicAddition();
    TestZeroVector();
    TestNegativeValues();
    TestLargeVector();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
