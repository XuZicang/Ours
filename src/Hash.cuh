#ifndef _HASH_CUH
#define _HASH_CUH
#include "Meta.h"

__forceinline__ __host__ __device__ uint32_t hash0(uintV id)
{
    uint32_t x = id;
    x = (x ^ (x >> 16)) * 0x7feb352dU;
    x = (x ^ (x >> 15)) * 0x846ca68bU;
    return x ^ (x >> 16);
}

__forceinline__ __host__ __device__ uint32_t hash1(uintV id)
{
    uint32_t x = id;
    x = ((x >> 17) | (x << (32 - 17))) ^ 0x85ebca6bU; // 循环右移17位
    x = (x ^ (x >> 16)) * 0xcc9e2d51U;
    x = (x ^ (x >> 17)) * 0x1b873593U;
    return x;
}
#endif