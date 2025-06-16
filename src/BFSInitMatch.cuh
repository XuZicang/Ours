#pragma once
#include "Meta.h"
#include <cuda.h>
#include "utils/GPUUtil.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "Matchutil.cuh"
using namespace cooperative_groups;

__launch_bounds__(1024)
    __global__ void InitMatch(
        uintV *init_match,
        uintV vertex_count)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    for (; index < vertex_count; index += blockDim.x * gridDim.x)
        init_match[index] = index;
}

__launch_bounds__(1024, 1)
    __global__ void BFSKernelCompute(
        uintV vertex_count,
        uintE *row_ptrs,
        uintV *cols,
        uint32_t max_degree,
        uint8_t q_vertex_count,
        const uint32_t *__restrict__ q_row_ptrs,
        const uint8_t *__restrict__ q_cols,
        const uint8_t *__restrict__ restriction,
        uint64_t old_num,
        uintV *old_match,
        uint8_t mapping_layer,
        uint64_t *offsets)
{
    uint32_t warpNum = gridDim.x * blockDim.x / 32;
    __shared__ uintV partial_match[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    uint8_t parent_end = q_row_ptrs[mapping_layer + 1];
    uint8_t parent_start = q_row_ptrs[mapping_layer];
    for (uint64_t i = warpId; i < old_num; i += warpNum)
    {
        if (threadIdinWarp == 0)
            offsets[i] = 0;
        if (threadIdinWarp < mapping_layer)
        {
            partial_match[warpIdinBlock * MAX_QUERY_VERTEX + threadIdinWarp] = old_match[i * mapping_layer + threadIdinWarp];
        }
        __syncwarp();
        uint8_t parent_start = q_row_ptrs[mapping_layer];
        uint8_t parent_num = q_row_ptrs[mapping_layer + 1] - parent_start;
        uintV pivot_parent = partial_match[warpIdinBlock * MAX_QUERY_VERTEX + q_cols[parent_start]];
        uintE neighbor_start = row_ptrs[pivot_parent];
        uintE neighbor_end = row_ptrs[pivot_parent + 1];
        for (uintE neighbor_idx = neighbor_start + threadIdinWarp; neighbor_idx < neighbor_end; neighbor_idx += 32)
        {
            bool flag = true;
            uintV vid = cols[neighbor_idx];
            for (uint8_t p = 0; p < mapping_layer; p++)
            {
                if (partial_match[warpIdinBlock * MAX_QUERY_VERTEX + p] == vid)
                    flag = false;
            }
            if (restriction[mapping_layer] != 0xFF)
            {
                flag = flag && (vid > partial_match[warpIdinBlock * MAX_QUERY_VERTEX + restriction[mapping_layer]]);
            }
            for (uint8_t p = parent_start + 1; p < parent_end; p++)
            {
                flag = flag && BinarySearch(vid, cols + row_ptrs[partial_match[warpIdinBlock * MAX_QUERY_VERTEX + q_cols[p]]], row_ptrs[partial_match[warpIdinBlock * MAX_QUERY_VERTEX + q_cols[p]] + 1] - row_ptrs[partial_match[warpIdinBlock * MAX_QUERY_VERTEX + q_cols[p]]]);
            }
            __syncwarp(__activemask());
            uint64_t match_num = __reduce_add_sync(__activemask(), flag);
            if (threadIdinWarp == 0)
            {
                offsets[i] += match_num;
            }
            __syncwarp(__activemask());
        }
        __syncwarp();
    }
}

__device__ uint64_t ScanWarp(uint64_t val)
{
    uint64_t tmp;
    for (int offset = 1; offset < 32; offset *= 2)
    {
        tmp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdinWarp >= offset)
        {
            val += tmp;
        }
    }

    return val;
}
__device__ void ScanBlock(uint64_t *shm_data, uint64_t *warp_sum)
{
    *shm_data = ScanWarp(*shm_data);
    __syncthreads();
    if (threadIdinWarp == 31)
    {
        warp_sum[warpIdinBlock] = *shm_data;
    }
    __syncthreads();
    if (warpIdinBlock == 0)
    {
        warp_sum[threadIdinWarp] = ScanWarp(warp_sum[threadIdinWarp]);
    }
    __syncthreads();
    if (warpIdinBlock)
    {
        *shm_data += warp_sum[warpIdinBlock - 1];
    }
    __syncthreads();
}

__launch_bounds__(1024)
    __global__ void PartSum(uint64_t *offsets, uint64_t num, uint64_t *part, uint64_t partnum)
{
    __shared__ uint64_t shm[WARP_PER_BLOCK * 32 + WARP_PER_BLOCK];
    for (uint64_t i = blockIdx.x; i < partnum; i += gridDim.x)
    {
        uint64_t index = i * blockDim.x + threadIdx.x;
        shm[WARP_PER_BLOCK + threadIdx.x] = index < num ? offsets[index] : 0;
        __syncthreads();
        ScanBlock(shm + WARP_PER_BLOCK + threadIdx.x, shm);
        __syncthreads();
        if (index < num)
        {
            offsets[index] = shm[WARP_PER_BLOCK + threadIdx.x];
        }
        if (threadIdx.x == blockDim.x - 1)
        {
            part[i] = shm[WARP_PER_BLOCK + threadIdx.x];
        }
    }
}

__launch_bounds__(1024)
    __global__ void ScanPartSumKernel(uint64_t *part, uint64_t part_num)
{
    uint64_t acc = 0;
    for (uint64_t i = 0; i < part_num; i++)
    {
        acc += part[i];
        part[i] = acc;
    }
}

__launch_bounds__(1024)
    __global__ void AddBaseSumKernel(uint64_t *part, uint64_t *offset, uint64_t num, uint64_t part_num)
{
    for (uint64_t i = blockIdx.x; i < part_num; i += gridDim.x)
    {
        if (i == 0)
        {
            continue;
        }
        uint64_t index = i * blockDim.x + threadIdx.x;
        if (index < num)
            offset[index] += part[i - 1];
    }
}

void PrefixSum(uint64_t *offsets, uint64_t num, uint64_t *buffer, cudaStream_t stream)
{
    uint64_t part_size = 1024;
    uint64_t part_num = (num + part_size - 1) / part_size;
    uint64_t block_num = part_num;
    uint64_t *part = buffer;
    PartSum<<<block_num, 1024, 0, stream>>>(offsets, num, buffer, part_num);
    CUDA_ERROR(cudaStreamSynchronize(stream));
    if (part_num >= 2)
    {
        PrefixSum(part, part_num, buffer + part_num, stream);
        AddBaseSumKernel<<<block_num, part_size, 0, stream>>>(part, offsets, num, part_num);
        CUDA_ERROR(cudaStreamSynchronize(stream));
    }
}

__launch_bounds__(1024)
    __global__ void BFSKernelStore(
        uintV vertex_count,
        uintE *row_ptrs,
        uintV *cols,
        uint32_t max_degree,
        uint8_t q_vertex_count,
        const uint32_t *__restrict__ q_row_ptrs,
        const uint8_t *__restrict__ q_cols,
        const uint8_t *__restrict__ restriction,
        uint64_t old_num,
        uintV *old_match,
        uint8_t mapping_layer,
        uint64_t *offsets,
        uintV *new_match)
{
    uint32_t warpNum = gridDim.x * blockDim.x / 32;
    __shared__ uintV partial_match[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    uint8_t parent_end = q_row_ptrs[mapping_layer + 1];
    uint8_t parent_start = q_row_ptrs[mapping_layer];
    for (uint64_t i = warpId; i < old_num; i += warpNum)
    {
        if (threadIdinWarp < mapping_layer)
        {
            partial_match[warpIdinBlock * MAX_QUERY_VERTEX + threadIdinWarp] = old_match[i * mapping_layer + threadIdinWarp];
        }
        __syncwarp();
        uint8_t parent_start = q_row_ptrs[mapping_layer];
        uint8_t parent_num = q_row_ptrs[mapping_layer + 1] - parent_start;
        uintV pivot_parent = partial_match[warpIdinBlock * MAX_QUERY_VERTEX + q_cols[parent_start]];
        uintE neighbor_start = row_ptrs[pivot_parent];
        uintE neighbor_end = row_ptrs[pivot_parent + 1];
        uint32_t match_num = 0;
        for (uintE neighbor_idx = neighbor_start + threadIdinWarp; neighbor_idx < neighbor_end; neighbor_idx += 32)
        {
            bool flag = true;
            uintV vid = cols[neighbor_idx];
            for (uint8_t p = 0; p < mapping_layer; p++)
            {
                if (partial_match[warpIdinBlock * MAX_QUERY_VERTEX + p] == vid)
                    flag = false;
            }
            if (restriction[mapping_layer] != 0xFF)
            {
                flag = flag && (vid > partial_match[warpIdinBlock * MAX_QUERY_VERTEX + restriction[mapping_layer]]);
            }
            for (uint8_t p = parent_start + 1; p < parent_end; p++)
            {
                flag = flag && BinarySearch(vid, cols + row_ptrs[partial_match[warpIdinBlock * MAX_QUERY_VERTEX + q_cols[p]]], row_ptrs[partial_match[warpIdinBlock * MAX_QUERY_VERTEX + q_cols[p]] + 1] - row_ptrs[partial_match[warpIdinBlock * MAX_QUERY_VERTEX + q_cols[p]]]);
            }
            __syncwarp(__activemask());
            if (flag)
            {
                coalesced_group active = cooperative_groups::coalesced_threads();
                uint64_t index = offsets[i] + match_num + active.thread_rank();
                for (uint8_t j = 0; j < mapping_layer; j++)
                {
                    new_match[index * (mapping_layer + 1) + j] = partial_match[warpIdinBlock * MAX_QUERY_VERTEX + j];
                }
                new_match[index * (mapping_layer + 1) + mapping_layer] = vid;
            }
            match_num += __reduce_add_sync(__activemask(), flag);
            __syncwarp(__activemask());
        }
        __syncwarp();
    }
}
