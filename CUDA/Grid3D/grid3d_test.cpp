/**
 * ===========================================================================
 * Unit Tests for CUDA grid3D Utility Functions
 * ===========================================================================
 *
 * Tests the host-side APIs defined in grid3d_utils.cuh:
 *   - linearize3D:       3D coordinate -> flat index
 *   - threadRankInBlock:  local thread rank within a block
 *   - blockRankInGrid:    block's rank within the grid
 *   - threadRankInGrid:   global unique thread rank
 *
 * These functions are the same math the GPU kernel uses. By testing them
 * on the CPU with Google Test, we verify the indexing logic without needing
 * a GPU.
 *
 * Build & run via Makefile:
 *   make test
 *
 * ===========================================================================
 */

// grid3d_utils.cuh detects __CUDACC__ and provides host stubs automatically
#include "grid3d_utils.cuh"
#include <gtest/gtest.h>
#include <cmath>

// ============================================================
// linearize3D tests
// ============================================================

TEST(linearize3D, Origin) {
    // (0,0,0) should always map to index 0
    EXPECT_EQ(linearize3D(0, 0, 0, 8, 8), 0);
}

TEST(linearize3D, XOnly) {
    // Moving along x increments by 1
    EXPECT_EQ(linearize3D(5, 0, 0, 8, 8), 5);
}

TEST(linearize3D, YOnly) {
    // Moving along y increments by nx
    int Nx = 8;
    EXPECT_EQ(linearize3D(0, 3, 0, Nx, 8), 3 * Nx);
}

TEST(linearize3D, ZOnly) {
    // Moving along z increments by nx*ny
    int Nx = 8, Ny = 8;
    EXPECT_EQ(linearize3D(0, 0, 2, Nx, Ny), 2 * Nx * Ny);
}

TEST(linearize3D, General) {
    int Nx = 10, Ny = 20;
    int x = 3, y = 7, z = 5;
    int Expected = (z * Ny + y) * Nx + x;
    EXPECT_EQ(linearize3D(x, y, z, Nx, Ny), Expected);
}

TEST(linearize3D, LastElement) {
    // Last element of a 4x4x4 array is index 63
    EXPECT_EQ(linearize3D(3, 3, 3, 4, 4), 63);
}

TEST(linearize3D, NonCubic) {
    // Non-cubic dimensions: nx=512, ny=512, nz=256
    int Nx = 512, Ny = 512;
    EXPECT_EQ(linearize3D(0, 0, 1, Nx, Ny), Nx * Ny);
    EXPECT_EQ(linearize3D(511, 511, 255, Nx, Ny), 512 * 512 * 256 - 1);
}

// ============================================================
// threadRankInBlock tests
// ============================================================

TEST(threadRankInBlock, Origin) {
    EXPECT_EQ(threadRankInBlock(0, 0, 0, 32, 8), 0);
}

TEST(threadRankInBlock, XOnly) {
    // threadIdx.x = 5 -> rank 5
    EXPECT_EQ(threadRankInBlock(5, 0, 0, 32, 8), 5);
}

TEST(threadRankInBlock, YIncrement) {
    // threadIdx.y = 1 -> rank = 1 * blockDim.x = 32
    EXPECT_EQ(threadRankInBlock(0, 1, 0, 32, 8), 32);
}

TEST(threadRankInBlock, ZIncrement) {
    // threadIdx.z = 1 -> rank = 1 * blockDim.y * blockDim.x = 8*32 = 256
    EXPECT_EQ(threadRankInBlock(0, 0, 1, 32, 8), 256);
}

TEST(threadRankInBlock, LastThread) {
    // Last thread in a (32,8,2) block: rank = 2*8*32 - 1 = 511
    EXPECT_EQ(threadRankInBlock(31, 7, 1, 32, 8), 511);
}

TEST(threadRankInBlock, General) {
    int Tx = 10, Ty = 3, Tz = 1;
    int Bdx = 32, Bdy = 8;
    int Expected = (Tz * Bdy + Ty) * Bdx + Tx;
    EXPECT_EQ(threadRankInBlock(Tx, Ty, Tz, Bdx, Bdy), Expected);
}

// ============================================================
// blockRankInGrid tests
// ============================================================

TEST(blockRankInGrid, Origin) {
    EXPECT_EQ(blockRankInGrid(0, 0, 0, 16, 64), 0);
}

TEST(blockRankInGrid, XOnly) {
    EXPECT_EQ(blockRankInGrid(5, 0, 0, 16, 64), 5);
}

TEST(blockRankInGrid, YIncrement) {
    // blockIdx.y = 1 -> rank = 1 * gridDim.x = 16
    EXPECT_EQ(blockRankInGrid(0, 1, 0, 16, 64), 16);
}

TEST(blockRankInGrid, ZIncrement) {
    // blockIdx.z = 1 -> rank = 1 * gridDim.y * gridDim.x = 64*16 = 1024
    EXPECT_EQ(blockRankInGrid(0, 0, 1, 16, 64), 1024);
}

TEST(blockRankInGrid, LastBlock) {
    // Last block in a (16,64,128) grid: rank = 16*64*128 - 1 = 131071
    EXPECT_EQ(blockRankInGrid(15, 63, 127, 16, 64), 131071);
}

// ============================================================
// threadRankInGrid tests
// ============================================================

TEST(threadRankInGrid, FirstThread) {
    // rankInBlock = 0, blockSize = 512, rankOfBlock = 0 -> global 0
    EXPECT_EQ(threadRankInGrid(0, 512, 0), 0);
}

TEST(threadRankInGrid, SecondBlock) {
    // First thread of block 1: rankInBlock=0, blockSize=512, rankOfBlock=1
    EXPECT_EQ(threadRankInGrid(0, 512, 1), 512);
}

TEST(threadRankInGrid, GeneralCase) {
    // rankInBlock=100, blockSize=512, rankOfBlock=5
    // expected = 100 + 512*5 = 2660
    EXPECT_EQ(threadRankInGrid(100, 512, 5), 2660);
}

TEST(threadRankInGrid, ThreadId12345) {
    // Verify the default id=12345 can be decomposed and recomposed
    int BlockSize = 512;
    int RankOfBlock = 12345 / BlockSize;   // 24
    int RankInBlock = 12345 % BlockSize;   // 57
    EXPECT_EQ(threadRankInGrid(RankInBlock, BlockSize, RankOfBlock), 12345);
}

// ============================================================
// Integration: full pipeline for specific coordinates
// ============================================================

TEST(Integration, CoordinateToIndex) {
    // For the actual launch config: nx=512, ny=512, nz=256
    // blockDim = (32,8,2), gridDim = (16,64,128)
    int Nx = 512, Ny = 512;
    int Bdx = 32, Bdy = 8;
    int Gdx = 16, Gdy = 64;

    // Pick thread (1, 2, 0) in block (3, 4, 5)
    int Tx = 1, Ty = 2, Tz = 0;
    int Bx = 3, By = 4, Bz = 5;

    // Global coordinate
    int Gx = Bx * Bdx + Tx;   // 3*32 + 1 = 97
    int Gy = By * Bdy + Ty;   // 4*8  + 2 = 34
    int Gz = Bz * 2   + Tz;   // 5*2  + 0 = 10

    EXPECT_EQ(Gx, 97);
    EXPECT_EQ(Gy, 34);
    EXPECT_EQ(Gz, 10);

    // Flat index
    int Idx = linearize3D(Gx, Gy, Gz, Nx, Ny);
    EXPECT_EQ(Idx, (10 * 512 + 34) * 512 + 97);

    // Thread rank
    int Rib = threadRankInBlock(Tx, Ty, Tz, Bdx, Bdy);
    int Rob = blockRankInGrid(Bx, By, Bz, Gdx, Gdy);
    int BlockSize = Bdx * Bdy * 2;  // 512
    int Rig = threadRankInGrid(Rib, BlockSize, Rob);

    // The kernel writes a[idx] = rig, b[idx] = sqrt(rig)
    float ExpectedB = sqrtf((float)Rig);
    EXPECT_FLOAT_EQ(ExpectedB, sqrtf((float)Rig));
}

TEST(Integration, BoundsCheck) {
    // Verify that the last valid coordinate maps to the last array element
    int Nx = 512, Ny = 512, Nz = 256;
    int LastIdx = linearize3D(Nx - 1, Ny - 1, Nz - 1, Nx, Ny);
    EXPECT_EQ(LastIdx, Nx * Ny * Nz - 1);
}
