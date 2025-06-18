#pragma once
#include "Meta.h"
#include <cuda.h>
#include "utils/GPUUtil.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;

__device__ void printArray(const uint32_t *array, uint32_t size)
{
    if (threadIdinWarp == 0)
        printf("size: %d\n", size);
    __syncwarp();
    for (int i = threadIdinWarp; i < (size); i += 32)
    {
        printf("%d\n", array[i]);
    }
    __syncwarp();
}
__forceinline__ __device__ void load_init_match(uintE &init_match_pointer, uintE *allocator)
{
    if (threadIdinWarp == 0)
    {
        init_match_pointer = atomicAdd((unsigned long long *)allocator, 1UL);
    }
    init_match_pointer = __shfl_sync(0xFFFFFFFF, init_match_pointer, 0);
    return;
}

__forceinline__ __device__ bool BinarySearch(uintV vid, const uintV *array, uintV arraySize)
{
    if (arraySize == 0)
        return false;
    int l = 0, r = arraySize - 1;
    int mid;
    while (l <= r)
    {
        mid = (l + r) / 2;
        if (vid == array[mid])
        {
            return true;
            break;
        }
        else if (vid < array[mid])
            r = mid - 1;
        else
            l = mid + 1;
    }
    return false;
}

__forceinline__ __device__ int BinarySearchPosition(uintV vid, uintV *array, uintV arraySize)
{
    if (arraySize == 0)
        return -1;
    int l = 0, r = arraySize - 1;
    int mid;
    int result = -1;
    while (l <= r)
    {
        mid = (l + r) / 2;
        if (vid == array[mid])
        {
            result = mid;
            break;
        }
        else if (vid < array[mid])
            r = mid - 1;
        else
            l = mid + 1;
    }
    return result;
}

__forceinline__ __device__ void BinaryIntersectionUpperbound(
    uintV* arrayA,
    uintV* arrayB,
    uintV* writeArray,
    uintV sizeA,
    uintV sizeB,
    uintV& result_size,
    uintV upperbound
)
{
    result_size = 0;
    for (uint32_t i = threadIdinWarp; i < sizeA; i += 32)
    {
        uintV v = arrayA[i];
        if (v >= upperbound) break;
        bool exist = BinarySearch(v, arrayB, sizeB);
        if (exist)
        {
            coalesced_group active = cooperative_groups::coalesced_threads();
            writeArray[result_size + active.thread_rank()] = arrayA[i];
        }
        result_size += __reduce_add_sync(__activemask(), exist);
    }
    return;
}

__forceinline__ __device__ void BinaryExclusionUpperbound(uintV* arrayA, const uintV* arrayB, uintV* writeArray, uintV sizeA, uintV sizeB, uintV& result_size, uintV upperbound)
{
    result_size = 0;
    if (warpId == 0) {
    printArray(arrayA, sizeA);
    printArray(arrayB, sizeB);
    }
    for (uint32_t i = threadIdinWarp; i < sizeA; i += 32)
    {
        uintV v = arrayA[i];
        if (v >= upperbound) break;
        bool not_exist = !BinarySearch(v, arrayB, sizeB);
        // if (warpId == 0)
        //     printf("%d %d\n", v, not_exist);
        if (not_exist)
        {
            coalesced_group active = cooperative_groups::coalesced_threads();
            writeArray[result_size + active.thread_rank()] = arrayA[i];
        }
        result_size += __reduce_add_sync(__activemask(), not_exist);
    }
    if (warpId == 0)
    {
        printArray(writeArray, result_size);
    }
    return;
}

__forceinline__ __device__ void BinaryIntersection(uintV *arrayA, uintV *arrayB, uintV &sizeA, uintV sizeB)
{
    uintV write_pos = 0;
    for (uintV i = threadIdinWarp; i < sizeA; i += 32)
    {
        bool is_exist = BinarySearch(arrayA[i], arrayB, sizeB);
        if (is_exist)
        {
            coalesced_group active = cooperative_groups::coalesced_threads();
            uintV wptr = write_pos + active.thread_rank();
            arrayA[wptr] = arrayA[i];
        }
        write_pos += __reduce_add_sync(__activemask(), is_exist);
    }
    if (threadIdinWarp == 0)
        sizeA = write_pos;
    return;
}

__forceinline__ __device__ void BinaryIntersection(uintV *arrayA, uintV *arrayB, uintV* result, uintV &sizeA, uintV sizeB)
{
    uintV write_pos = 0;
    for (uintV i = threadIdinWarp; i < sizeA; i += 32)
    {
        bool is_exist = BinarySearch(arrayA[i], arrayB, sizeB);
        if (is_exist)
        {
            coalesced_group active = cooperative_groups::coalesced_threads();
            uintV wptr = write_pos + active.thread_rank();
            result[wptr] = arrayA[i];
        }
        write_pos += __reduce_add_sync(__activemask(), is_exist);
    }
    if (threadIdinWarp == 0)
        sizeA = write_pos;
    return;
}

__forceinline__ __device__ void ComputeInducedGraph(
    uintV vertex_count,
    uintE *row_ptrs,
    uintV *cols,
    uint32_t layer_size,
    uint32_t *Gind,
    uint32_t *Vorient,
    uintV vid,
    uint32_t &indSize)
{
    uintE start = row_ptrs[vid];
    indSize = row_ptrs[vid + 1] - start;
    for (uint32_t i = warpIdinBlock; i < indSize; i += WARP_PER_BLOCK)
    {
        for (uint32_t j = threadIdinWarp; j < layer_size; j += 32)
            Gind[i * layer_size + j] = 0;
    }
    for (uint32_t i = threadIdx.x; i < layer_size; i += blockDim.x)
    {
        Vorient[i] = 0;
    }
    __syncthreads();
    for (uintV i = warpIdinBlock; i < indSize; i += WARP_PER_BLOCK)
    {
        uintV v = cols[start + i];
        if (v < vid)
        {
            if (threadIdinWarp == 0)
                atomicOr(Vorient + i / 32, (1 << i % 32));
        }
        uintE s = row_ptrs[v];
        uintE e = row_ptrs[v + 1] - s;
        for (uintV j = threadIdinWarp; j < e; j += 32)
        {
            int pos = BinarySearchPosition(cols[s + j], cols + start, indSize);
            if (pos != -1)
            {
                atomicOr(Gind + layer_size * i + pos / 32, (1 << pos % 32));
            }
        }
    }
    __syncthreads();
    return;
}

__forceinline__ __device__ void SetExclusiveBinary(uint32_t *set1, uint32_t *set2, uint32_t indSize)
{
    for (uint32_t i = threadIdinWarp; i < (indSize + 31) / 32; i += 32)
    {
        set1[i] = set1[i] | (~set2[i]);
    }
    __syncwarp();
}

__forceinline__ __device__ void SetIntersectionBinary(uint32_t *set1, uint32_t *set2, uint32_t indSize)
{
    for (uint32_t i = threadIdinWarp; i < (indSize + 31) / 32; i += 32)
    {
        set1[i] = set1[i] & set2[i];
    }
    __syncwarp();
}

__forceinline__ __device__ void init_stack_layer(uint32_t *stacklayer, uint32_t *match_binary, uint32_t indSize)
{
    for (uint32_t i = threadIdinWarp; i < (indSize) / 32; i += 32)
    {
        stacklayer[i] = 0xFFFFFFFFU;
    }
    stacklayer[indSize / 32] = (1 << indSize % 32) - 1;
    __syncwarp();
    SetExclusiveBinary(stacklayer, match_binary, indSize);
}
__forceinline__ __device__ void init_match_binary(uint32_t *match_binary, uintV *init_match, uint8_t init_layer, uintV *Vind, uint32_t indSize)
{
    for (int i = threadIdinWarp; i < (indSize + 1023) / 1024 * 32; i += 32)
        match_binary[i] = 0;
    if (threadIdinWarp + 1 < init_layer)
    {
        int pos = BinarySearchPosition(init_match[threadIdinWarp + 1], Vind, indSize);
        atomicOr(match_binary + pos / 32, (uint32_t)(1 << (pos % 32)));
    }
    __syncwarp();
}

__forceinline__ __device__ void CountOnesBinary(uint32_t *array, uint32_t bits, uint32_t &num)
{
    uint32_t local_count = 0;
    for (int i = threadIdinWarp; i < (bits + 31) / 32; i += 32)
    {
        local_count += __popc(array[i]);
    }
    for (int offset = 16; offset > 0; offset /= 2)
    {
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);
    }
    if (threadIdinWarp == 0)
        num = local_count;
    __syncwarp();
    return;
}

__forceinline__ __device__ int FindPosition(uint32_t *array, uint32_t bits, uint32_t target)
{
    uint32_t count;
    uint32_t sum;
    __shared__ uint32_t index;
    __shared__ uint32_t share_prefix;
    __shared__ uint32_t share_item;
    index = 0xFFFFFFFFU;
    for (uint32_t i = threadIdinWarp; i < (bits + 31) / 32; i += 32)
    {
        count = __popc(array[i]);
        sum = count;
        for (int offset = 1; offset < 32; offset *= 2)
        {
            uint32_t temp = __shfl_up_sync(0xFFFFFFFF, sum, offset);
            if (threadIdinWarp > offset)
                sum += temp;
        }
        uint32_t prefix = sum - count;
        if (prefix <= target && target < sum)
        {
            index = i;
            share_prefix = prefix;
            share_item = array[i];
        }
        __syncwarp();
        if (index != 0xFFFFFFFFU)
            break;
    }
    uint32_t is_one = ((share_item & (1 << threadIdinWarp)) != 0);
    for (int offset = 1; offset < 32; offset *= 2)
    {
        uint32_t temp = __shfl_up_sync(0xFFFFFFFF, is_one, offset);
        if (threadIdinWarp > offset)
            is_one += temp;
    }
    bool cond = ((share_prefix + is_one) == target);
    unsigned mask = __ballot_sync(0xFFFFFFFFU, cond);
    if (mask == 0)
        return -1;
    else
    {
        return __ffs(mask) - 1 + index * 32;
    }
}

__forceinline__ __device__ void SetIntersectionBinary(uint32_t *set1, uint32_t *set2, uint32_t *result_set, uint32_t indSize)
{
    for (uint32_t i = threadIdinWarp; i < (indSize + 31) / 32; i += 32)
    {
        result_set[i] = set1[i] & set2[i];
    }
    __syncwarp();
}
__forceinline__ __device__ void FindFirstOneBinary(uint32_t *bits, uint32_t indSize, uint32_t pos, uint32_t &res)
{
    res = 0xFFFFFFFFU;
    uint32_t startWord = pos / 32;
    uint32_t startBit = pos % 32;
    uint32_t totalWords = (indSize + 31) / 32;
    for (uint32_t wordIdx = startWord + threadIdinWarp; wordIdx < totalWords; wordIdx += 32)
    {
        uint32_t word = bits[wordIdx];
        if (wordIdx == startWord)
        {
            word &= 0xFFFFFFFF << startBit;
        }
        uint32_t ballot = __ballot_sync(__activemask(), word != 0);
        if (ballot != 0)
        {
            int leaderLane = __ffs(ballot) - 1;
            if (threadIdinWarp == leaderLane)
            {
                int firstSet = __ffs(word) - 1;
                res = wordIdx * 32 + firstSet;
            }
        }
    }
    __syncwarp();
    return;
}

