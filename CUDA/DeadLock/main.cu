/**
 * @file main.cu
 * @brief Driver for the CUDA deadlock demonstration.
 *
 * Usage:
 *   ./main [warps] [blocks] [gsync] [dolock]
 *   Defaults: 3 warps, 1 block, gsync=32 (1 warp), dolock=1
 */

#include "DeadLock.cuh"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    int warps  = (argc > 1) ? atoi(argv[1]) : 3;
    int blocks = (argc > 2) ? atoi(argv[2]) : 1;
    int gsync  = (argc > 3) ? atoi(argv[3]) : 32;
    int doLock = (argc > 4) ? atoi(argv[4]) : 1;

    printf("Launching DeadLockKernel: %d blocks, %d threads/block, gsync=%d, doLock=%d\n",
           blocks, warps * 32, gsync, doLock);

    DeadLockKernel<<<blocks, warps * 32>>>(gsync, doLock);
    cudaDeviceSynchronize();

    printf("done\n");
    return 0;
}
