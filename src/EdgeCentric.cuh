#pragma once
#include "Meta.h"
#include <cuda.h>
#include "utils/GPUUtil.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "Balance.cuh"
#include "Matchutil.cuh"
using namespace cooperative_groups;

__forceinline__ __device__ void extend(
    uintV vertex_count,
    uintE *row_ptrs,
    uintV *cols,
    uint32_t layer_size,
    uintV *BFSlayer,
    uintV *stacks,
    uint8_t q_vertex_count,
    const uint32_t *__restrict__ q_row_ptrs,
    const uint8_t *__restrict__ q_cols,
    const uint8_t *__restrict__ restriction,
    uint8_t init_vertex_num,
    uintV *partial_match,
    uint32_t *stack_iter,
    uint32_t *stack_size,
    uint64_t &warpCounter,
    uint32_t &BFS_layer_size,
    uint64_t *Counter)
{
    uint8_t current_mapping_vertex = init_vertex_num;
    uint8_t parent_start = q_row_ptrs[current_mapping_vertex];
    uint8_t parent_num = q_row_ptrs[current_mapping_vertex + 1] - parent_start;
    uintV pivot_vertex = partial_match[q_cols[parent_start]];
    if (threadIdx.x == 0)
        BFS_layer_size = 0;
    __syncthreads();
    for (uintV i = row_ptrs[pivot_vertex] + threadIdx.x; i < row_ptrs[pivot_vertex + 1]; i += blockDim.x)
    {
        bool flag = true;
        uintV candidate = cols[i];
        if (restriction[current_mapping_vertex] != 0xFF && candidate <= partial_match[restriction[current_mapping_vertex]])
        {
            flag = false;
        }
        for (uint8_t j = 0; j < current_mapping_vertex; j++)
        {
            if (candidate == partial_match[j])
            {
                flag = false;
            }
        }
        if (parent_num > 1 && flag)
        {
            uintV parent = partial_match[q_cols[parent_start + 1]];
            if (!BinarySearch(candidate, cols + row_ptrs[parent],
                              row_ptrs[parent + 1] - row_ptrs[parent]))
            {
                flag = false;
            }
        }
        __syncwarp(__activemask());
        uint32_t wptr_base;
        uint32_t candidate_num = __reduce_add_sync(__activemask(), flag);
        if (threadIdinWarp == 0)
            wptr_base = atomicAdd(&BFS_layer_size, candidate_num);
        wptr_base = __shfl_sync(__activemask(), wptr_base, 0);
        if (flag)
        {
            coalesced_group active = cooperative_groups::coalesced_threads();
            BFSlayer[wptr_base + active.thread_rank()] = candidate;
        }
        __syncwarp(__activemask());
    }
    __syncthreads();
    if (current_mapping_vertex == q_vertex_count - 1)
    {
        if (threadIdx.x == 0)
        {
            atomicAdd((unsigned long long *)Counter, BFS_layer_size);
        }
        __syncthreads();
        return;
    }
    uint8_t init_layer = init_vertex_num + 1;
    for (uint32_t subtask_id = warpIdinBlock; subtask_id < BFS_layer_size; subtask_id += WARP_PER_BLOCK)
    {
        if (threadIdinWarp == 0)
        {
            partial_match[current_mapping_vertex] = BFSlayer[subtask_id];
        }
        __syncwarp();
        current_mapping_vertex++;
        while (true)
        {
            __syncwarp();
            uintV *stack_array = stacks + (current_mapping_vertex - init_layer) * layer_size;
            uint8_t parent_start = q_row_ptrs[current_mapping_vertex];
            uint8_t parent_num = q_row_ptrs[current_mapping_vertex + 1] - parent_start;
            if (restriction[current_mapping_vertex] != 0xFF)
            {
                uintV restrict_vid = partial_match[restriction[current_mapping_vertex]];
                uintV pivot_vid = partial_match[q_cols[parent_start]];
                uintV write_pos = 0;
                for (uintE i = row_ptrs[pivot_vid] + threadIdinWarp; i < row_ptrs[pivot_vid + 1]; i += 32)
                {
                    bool flag = true;
                    if (cols[i] < restrict_vid)
                    {
                        flag = false;
                    }
                    if (flag)
                        for (uint8_t j = 0; j < current_mapping_vertex; j++)
                        {
                            if (cols[i] == partial_match[j])
                            {
                                flag = false;
                                break;
                            }
                        }
                    __syncwarp(__activemask());
                    if (flag)
                    {
                        coalesced_group active = cooperative_groups::coalesced_threads();
                        uintV wptr = write_pos + active.thread_rank();
                        stack_array[wptr] = cols[i];
                    }
                    write_pos += __reduce_add_sync(__activemask(), flag);
                }
                write_pos = __shfl_sync(0xFFFFFFFF, write_pos, 0);
                __syncwarp();
                stack_size[current_mapping_vertex - init_layer] = __shfl_sync(0xFFFFFFFF, write_pos, 0);
            }
            else
            {
                uintV pivot_vid = partial_match[q_cols[parent_start]];
                uintV write_pos = 0;
                for (uintE i = row_ptrs[pivot_vid] + threadIdinWarp; i < row_ptrs[pivot_vid + 1]; i += 32)
                {
                    bool flag = true;
                    for (uint8_t j = 0; j < current_mapping_vertex; j++)
                    {
                        if (cols[i] == partial_match[j])
                        {
                            flag = false;
                            break;
                        }
                    }
                    __syncwarp(__activemask());
                    if (flag)
                    {
                        coalesced_group active = cooperative_groups::coalesced_threads();
                        uintV wptr = write_pos + active.thread_rank();
                        stack_array[wptr] = cols[i];
                    }
                    write_pos += __reduce_add_sync(__activemask(), flag);
                }
                stack_size[current_mapping_vertex - init_layer] = __shfl_sync(0xFFFFFFFF, write_pos, 0);
            }
            for (uint8_t parent = parent_start + 1; parent < parent_start + parent_num; parent++)
            {
                uintV parent_vid = partial_match[q_cols[parent]];
                BinaryIntersection(stack_array, cols + row_ptrs[parent_vid], stack_size[current_mapping_vertex - init_layer], row_ptrs[parent_vid + 1] - row_ptrs[parent_vid]);
                __syncwarp();
            }
            if (current_mapping_vertex == q_vertex_count - 1 || stack_size[current_mapping_vertex - init_layer] == 0)
            {
                if (threadIdinWarp == 0)
                {
                    warpCounter += stack_size[current_mapping_vertex - init_layer];
                }
                __syncwarp();
                current_mapping_vertex--;
                while (current_mapping_vertex >= init_layer && stack_iter[current_mapping_vertex - init_layer] == stack_size[current_mapping_vertex - init_layer] - 1)
                    current_mapping_vertex--;
                if (current_mapping_vertex == init_layer - 1)
                    break;
                stack_iter[current_mapping_vertex - init_layer]++;
                partial_match[current_mapping_vertex] =
                    stacks[(current_mapping_vertex - init_layer) * layer_size +
                           stack_iter[current_mapping_vertex - init_layer]];
                current_mapping_vertex++;
            }
            else
            {
                stack_iter[current_mapping_vertex - init_layer] = 0;
                partial_match[current_mapping_vertex] =
                    stacks[(current_mapping_vertex - init_layer) * layer_size];
                current_mapping_vertex++;
            }
        }
    }
    __syncthreads();
}

__forceinline__ __device__ void extendWithOffload(
    uintV vertex_count,
    uintE *row_ptrs,
    uintV *cols,
    uint32_t layer_size,
    uintV *BFSlayer,
    uintV *stacks,
    uint8_t q_vertex_count,
    const uint32_t *__restrict__ q_row_ptrs,
    const uint8_t *__restrict__ q_cols,
    const uint8_t *__restrict__ restriction,
    uint8_t init_vertex_num,
    uintV *partial_match,
    uint32_t *stack_iter,
    uint32_t *stack_size,
    uint64_t &warpCounter,
    uint32_t &BFS_layer_size,
    uint64_t *Counter,
    uint8_t offload_layer,
    WorkerQueue *queue)
{
    uint8_t current_mapping_vertex = init_vertex_num;
    uint8_t parent_start = q_row_ptrs[current_mapping_vertex];
    uint8_t parent_num = q_row_ptrs[current_mapping_vertex + 1] - parent_start;
    uintV pivot_vertex = partial_match[q_cols[parent_start]];
    if (threadIdx.x == 0)
        BFS_layer_size = 0;
    __syncthreads();
    for (uintV i = row_ptrs[pivot_vertex] + threadIdx.x; i < row_ptrs[pivot_vertex + 1]; i += blockDim.x)
    {
        bool flag = true;
        uintV candidate = cols[i];
        if (restriction[current_mapping_vertex] != 0xFF && candidate <= partial_match[restriction[current_mapping_vertex]])
        {
            flag = false;
        }
        for (uint8_t j = 0; j < current_mapping_vertex; j++)
        {
            if (candidate == partial_match[j])
            {
                flag = false;
                break;
            }
        }
        if (parent_num > 1 && flag)
        {
            for (uint8_t j = 1; j < parent_num; j++)
            {
                uintV parent = partial_match[q_cols[parent_start + j]];
                if (!BinarySearch(candidate, cols + row_ptrs[parent],
                                  row_ptrs[parent + 1] - row_ptrs[parent]))
                {
                    flag = false;
                    break;
                }
            }
        }
        __syncwarp(__activemask());
        uint32_t wptr_base;
        uint32_t candidate_num = __reduce_add_sync(__activemask(), flag);
        if (threadIdinWarp == 0)
            wptr_base = atomicAdd(&BFS_layer_size, candidate_num);
        wptr_base = __shfl_sync(__activemask(), wptr_base, 0);
        if (flag)
        {
            coalesced_group active = cooperative_groups::coalesced_threads();
            BFSlayer[wptr_base + active.thread_rank()] = candidate;
        }
        __syncwarp(__activemask());
    }
    __syncthreads();
    if (current_mapping_vertex == q_vertex_count - 1)
    {
        if (threadIdx.x == 0)
        {
            atomicAdd((unsigned long long *)Counter, BFS_layer_size);
        }
        __syncthreads();
        return;
    }
    uint8_t init_layer = init_vertex_num + 1;
    for (uint32_t subtask_id = warpIdinBlock; subtask_id < BFS_layer_size; subtask_id += WARP_PER_BLOCK)
    {
        if (threadIdinWarp == 0)
        {
            partial_match[current_mapping_vertex] = BFSlayer[subtask_id];
        }
        __syncwarp();
        current_mapping_vertex++;
        while (true)
        {
            if (current_mapping_vertex <= offload_layer && queue->queueCount)
            {
                if (inform(*queue, partial_match, current_mapping_vertex))
                {
                    __syncwarp();
                    current_mapping_vertex--;
                    while (current_mapping_vertex >= init_layer && stack_iter[current_mapping_vertex - init_layer] == stack_size[current_mapping_vertex - init_layer] - 1)
                        current_mapping_vertex--;
                    if (current_mapping_vertex == init_layer - 1)
                        break;
                    stack_iter[current_mapping_vertex - init_layer]++;
                    partial_match[current_mapping_vertex] =
                        stacks[(current_mapping_vertex - init_layer) * layer_size +
                               stack_iter[current_mapping_vertex - init_layer]];
                    current_mapping_vertex++;
                }
            }
            __syncwarp();
            uintV *stack_array = stacks + (current_mapping_vertex - init_layer) * layer_size;
            uint8_t parent_start = q_row_ptrs[current_mapping_vertex];
            uint8_t parent_num = q_row_ptrs[current_mapping_vertex + 1] - parent_start;
            if (restriction[current_mapping_vertex] != 0xFF)
            {
                uintV restrict_vid = partial_match[restriction[current_mapping_vertex]];
                uintV pivot_vid = partial_match[q_cols[parent_start]];
                uintV write_pos = 0;
                for (uintE i = row_ptrs[pivot_vid] + threadIdinWarp; i < row_ptrs[pivot_vid + 1]; i += 32)
                {
                    bool flag = true;
                    if (cols[i] < restrict_vid)
                    {
                        flag = false;
                    }
                    if (flag)
                        for (uint8_t j = 0; j < current_mapping_vertex; j++)
                        {
                            if (cols[i] == partial_match[j])
                            {
                                flag = false;
                                break;
                            }
                        }
                    __syncwarp(__activemask());
                    if (flag)
                    {
                        coalesced_group active = cooperative_groups::coalesced_threads();
                        uintV wptr = write_pos + active.thread_rank();
                        stack_array[wptr] = cols[i];
                    }
                    write_pos += __reduce_add_sync(__activemask(), flag);
                }
                write_pos = __shfl_sync(0xFFFFFFFF, write_pos, 0);
                __syncwarp();
                stack_size[current_mapping_vertex - init_layer] = __shfl_sync(0xFFFFFFFF, write_pos, 0);
            }
            else
            {
                uintV pivot_vid = partial_match[q_cols[parent_start]];
                uintV write_pos = 0;
                for (uintE i = row_ptrs[pivot_vid] + threadIdinWarp; i < row_ptrs[pivot_vid + 1]; i += 32)
                {
                    bool flag = true;
                    for (uint8_t j = 0; j < current_mapping_vertex; j++)
                    {
                        if (cols[i] == partial_match[j])
                        {
                            flag = false;
                            break;
                        }
                    }
                    __syncwarp(__activemask());
                    if (flag)
                    {
                        coalesced_group active = cooperative_groups::coalesced_threads();
                        uintV wptr = write_pos + active.thread_rank();
                        stack_array[wptr] = cols[i];
                    }
                    write_pos += __reduce_add_sync(__activemask(), flag);
                }
                stack_size[current_mapping_vertex - init_layer] = __shfl_sync(0xFFFFFFFF, write_pos, 0);
            }
            for (uint8_t parent = parent_start + 1; parent < parent_start + parent_num; parent++)
            {
                uintV parent_vid = partial_match[q_cols[parent]];
                BinaryIntersection(stack_array, cols + row_ptrs[parent_vid], stack_size[current_mapping_vertex - init_layer], row_ptrs[parent_vid + 1] - row_ptrs[parent_vid]);
                __syncwarp();
            }
            if (current_mapping_vertex == q_vertex_count - 1 || stack_size[current_mapping_vertex - init_layer] == 0)
            {
                __syncwarp();
                if (threadIdinWarp == 0)
                {
                    warpCounter += stack_size[current_mapping_vertex - init_layer];
                }
                __syncwarp();
                current_mapping_vertex--;
                while (current_mapping_vertex >= init_layer && stack_iter[current_mapping_vertex - init_layer] == stack_size[current_mapping_vertex - init_layer] - 1)
                    current_mapping_vertex--;
                if (current_mapping_vertex == init_layer - 1)
                    break;
                stack_iter[current_mapping_vertex - init_layer]++;
                partial_match[current_mapping_vertex] =
                    stacks[(current_mapping_vertex - init_layer) * layer_size +
                           stack_iter[current_mapping_vertex - init_layer]];
                current_mapping_vertex++;
            }
            else
            {
                stack_iter[current_mapping_vertex - init_layer] = 0;
                partial_match[current_mapping_vertex] =
                    stacks[(current_mapping_vertex - init_layer) * layer_size];
                current_mapping_vertex++;
            }
        }
    }
    __syncthreads();
}

__launch_bounds__(1024, 2)
    __global__ void EdgeCentricDFSBalancedKernel(
        uintV vertex_count,
        uintE *row_ptrs,
        uintV *cols,
        uint32_t layer_size,
        uintV *BFSlayer,
        uintV *stacks,
        uint8_t q_vertex_count,
        const uint32_t *__restrict__ q_row_ptrs,
        const uint8_t *__restrict__ q_cols,
        const uint8_t *__restrict__ restriction,
        uintE init_edge_num,
        uintV *init_edges,
        uint64_t *Counter,
        uint8_t offload_layer,
        WorkerQueue *queue)
{
    __shared__ uintV partial_match[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_iter[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_size[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint64_t warpCounter[WARP_PER_BLOCK];
#ifdef PROFILE
    __shared__ uint64_t total_clock_start;
    __shared__ uint64_t total_clocks;
    __shared__ uint64_t traverse_clock_start;
    __shared__ uint64_t traverse_clocks;
    __shared__ uint64_t wait_clock_start;
    __shared__ uint64_t wait_clocks;
#endif
    __shared__ uint32_t queuepos;
    __shared__ uint32_t BFS_layer_size;
    uintE init_match_pointer;
    uint8_t init_layer = 3;
    warpCounter[warpIdinBlock] = 0;
#ifdef PROFILE
    if (threadIdx.x == 0)
    {
        total_clocks = 0;
        traverse_clocks = 0;
        wait_clocks = 0;
        total_clock_start = clock64();
    }
#endif
    for (init_match_pointer = blockIdx.x; init_match_pointer < init_edge_num; init_match_pointer += gridDim.x)
    {
        uint8_t current_mapping_vertex = 2;
        if (threadIdinWarp < current_mapping_vertex)
            partial_match[warpIdinBlock * MAX_QUERY_VERTEX + threadIdinWarp] = init_edges[init_match_pointer * 2 + threadIdinWarp];
        __syncthreads();
#ifdef PROFILE
        if (threadIdx.x == 0)
        {
            traverse_clock_start = clock64();
        }
#endif
        extendWithOffload(
            vertex_count,
            row_ptrs,
            cols,
            layer_size,
            BFSlayer + blockIdx.x * layer_size,
            stacks + warpId * (q_vertex_count - init_layer) * layer_size,
            q_vertex_count,
            q_row_ptrs,
            q_cols,
            restriction,
            2,
            partial_match + warpIdinBlock * MAX_QUERY_VERTEX,
            stack_iter + warpIdinBlock * MAX_QUERY_VERTEX,
            stack_size + warpIdinBlock * MAX_QUERY_VERTEX,
            warpCounter[warpIdinBlock],
            BFS_layer_size,
            Counter,
            offload_layer,
            queue);
        __syncthreads();
#ifdef PROFILE
        if (threadIdx.x == 0)
        {
            traverse_clocks += clock64() - traverse_clock_start;
        }
#endif
    }
    __syncthreads();
    while (true)
    {
#ifdef PROFILE
        if (threadIdx.x == 0)
            wait_clock_start = clock64();
#endif
        enqueue(*queue, queuepos);
        uint32_t init_vertex_num;
        if (waitTask(*queue, queuepos, partial_match + warpIdinBlock * MAX_QUERY_VERTEX, init_vertex_num))
        {
            dequeue(*queue, queuepos);
#ifdef PROFILE
            if (threadIdx.x == 0)
            {
                wait_clocks += clock64() - wait_clock_start;
                traverse_clock_start = clock64();
            }
#endif
            extendWithOffload(
                vertex_count,
                row_ptrs,
                cols,
                layer_size,
                BFSlayer + blockIdx.x * layer_size,
                stacks + warpId * (q_vertex_count - init_layer) * layer_size,
                q_vertex_count,
                q_row_ptrs,
                q_cols,
                restriction,
                init_vertex_num,
                partial_match + warpIdinBlock * MAX_QUERY_VERTEX,
                stack_iter + warpIdinBlock * MAX_QUERY_VERTEX,
                stack_size + warpIdinBlock * MAX_QUERY_VERTEX,
                warpCounter[warpIdinBlock],
                BFS_layer_size,
                Counter,
                offload_layer,
                queue);
#ifdef PROFILE
            if (threadIdx.x == 0)
            {
                traverse_clocks += clock64() - traverse_clock_start;
            }
#endif
        }
        else
        {
#ifdef PROFILE
            if (threadIdx.x == 0)
                wait_clocks += clock64() - wait_clock_start;
#endif
            break;
        }
    }
#ifdef PROFILE
    if (threadIdx.x == 0)
        total_clocks += clock64() - total_clock_start;
#endif
#ifdef PROFILE
    if (threadIdx.x == 0)
        printf("blockId: %d, %lu, %lu, %lu\n", blockIdx.x, total_clocks, wait_clocks, traverse_clocks);
#endif
    if (threadIdinWarp == 0)
    {
        atomicAdd((unsigned long long *)Counter, (unsigned long long)warpCounter[warpIdinBlock]);
    }
}

__launch_bounds__(1024, 1)
    __global__ void EdgeCentricDFSKernel(
        uintV vertex_count,
        uintE *row_ptrs,
        uintV *cols,
        uint32_t layer_size,
        uintV *BFSlayer,
        uintV *stacks,
        uint8_t q_vertex_count,
        const uint32_t *__restrict__ q_row_ptrs,
        const uint8_t *__restrict__ q_cols,
        const uint8_t *__restrict__ restriction,
        uintE init_edge_num,
        uintV *init_edges,
        uint64_t *Counter)
{
    __shared__ uintV partial_match[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_iter[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_size[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint64_t warpCounter[WARP_PER_BLOCK];
#ifdef PROFILE
    __shared__ uint64_t total_clock_start;
    __shared__ uint64_t total_clocks;
#endif
    __shared__ uint32_t BFS_layer_size;
    uintE init_match_pointer;
    uint8_t init_layer = 3;
    warpCounter[warpIdinBlock] = 0;
#ifdef PROFILE
    if (threadIdx.x == 0)
    {
        total_clocks = 0;
        total_clock_start = clock64();
    }
#endif
    for (init_match_pointer = blockIdx.x; init_match_pointer < init_edge_num; init_match_pointer += gridDim.x)
    {
        uint8_t current_mapping_vertex = 2;
        if (threadIdinWarp < current_mapping_vertex)
            partial_match[warpIdinBlock * MAX_QUERY_VERTEX + threadIdinWarp] = init_edges[init_match_pointer * 2 + threadIdinWarp];
        __syncthreads();
        extend(
            vertex_count,
            row_ptrs,
            cols,
            layer_size,
            BFSlayer + blockIdx.x * layer_size,
            stacks + warpId * (q_vertex_count - init_layer) * layer_size,
            q_vertex_count,
            q_row_ptrs,
            q_cols,
            restriction,
            2,
            partial_match + warpIdinBlock * MAX_QUERY_VERTEX,
            stack_iter + warpIdinBlock * MAX_QUERY_VERTEX,
            stack_size + warpIdinBlock * MAX_QUERY_VERTEX,
            warpCounter[warpIdinBlock],
            BFS_layer_size,
            Counter);
        __syncthreads();
    }
#ifdef PROFILE
    if (threadIdx.x == 0)
        total_clocks += clock64() - total_clock_start;
#endif
#ifdef PROFILE
    if (threadIdx.x == 0)
        printf("blockId: %d, %lu\n", blockIdx.x, total_clocks);
#endif
    if (threadIdinWarp == 0)
    {
        atomicAdd((unsigned long long *)Counter, (unsigned long long)warpCounter[warpIdinBlock]);
    }
}

__forceinline__ __device__ void extendWithReuse(
    uintV vertex_count,
    uintE *row_ptrs,
    uintV *cols,
    uint32_t layer_size,
    uintV *BFSlayer,
    uintV *stacks,
    uintV *reuse_store,
    uint8_t q_vertex_count,
    const uint32_t *__restrict__ q_row_ptrs,
    const uint8_t *__restrict__ q_cols,
    const uint8_t *__restrict__ restriction,
    const uint8_t *__restrict__ reuse_pos,
    const uint8_t *__restrict__ reuse_q_cols,
    uint8_t init_vertex_num,
    uintV *partial_match,
    uint32_t *stack_iter,
    uint32_t *stack_size,
    uint32_t *reuse_size,
    uint64_t &warpCounter,
    uint32_t &BFS_layer_size,
    uint64_t *Counter)
{
    uint8_t current_mapping_vertex = init_vertex_num;
    uint8_t parent_start = q_row_ptrs[current_mapping_vertex];
    uint8_t parent_num = q_row_ptrs[current_mapping_vertex + 1] - parent_start;
    uintV pivot_vertex = partial_match[q_cols[parent_start]];
    if (threadIdx.x == 0)
        BFS_layer_size = 0;
    __syncthreads();
    for (uintV i = row_ptrs[pivot_vertex] + threadIdx.x; i < row_ptrs[pivot_vertex + 1]; i += blockDim.x)
    {
        bool flag = true;
        uintV candidate = cols[i];
        if (restriction[current_mapping_vertex] != 0xFF && candidate <= partial_match[restriction[current_mapping_vertex]])
        {
            flag = false;
        }
        for (uint8_t j = 0; j < current_mapping_vertex; j++)
        {
            if (candidate == partial_match[j])
            {
                flag = false;
            }
        }
        if (parent_num > 1 && flag)
        {
            uintV parent = partial_match[q_cols[parent_start + 1]];
            if (!BinarySearch(candidate, cols + row_ptrs[parent],
                              row_ptrs[parent + 1] - row_ptrs[parent]))
            {
                flag = false;
            }
        }
        __syncwarp(__activemask());
        uint32_t wptr_base;
        uint32_t candidate_num = __reduce_add_sync(__activemask(), flag);
        if (threadIdinWarp == 0)
            wptr_base = atomicAdd(&BFS_layer_size, candidate_num);
        wptr_base = __shfl_sync(__activemask(), wptr_base, 0);
        if (flag)
        {
            coalesced_group active = cooperative_groups::coalesced_threads();
            BFSlayer[wptr_base + active.thread_rank()] = candidate;
        }
        __syncwarp(__activemask());
    }
    __syncthreads();
    if (current_mapping_vertex == q_vertex_count - 1)
    {
        if (threadIdx.x == 0)
        {
            atomicAdd((unsigned long long *)Counter, BFS_layer_size);
        }
        __syncthreads();
        return;
    }
    uint8_t init_layer = init_vertex_num + 1;
    for (uint32_t subtask_id = warpIdinBlock; subtask_id < BFS_layer_size; subtask_id += WARP_PER_BLOCK)
    {
        if (threadIdinWarp == 0)
        {
            partial_match[current_mapping_vertex] = BFSlayer[subtask_id];
        }
        __syncwarp();
        current_mapping_vertex++;
        while (true)
        {
            __syncwarp();
            uintV *stack_array = stacks + (current_mapping_vertex)*layer_size;
            uintV *reuse_store_array = reuse_store + (current_mapping_vertex)*layer_size;
            uint8_t parent_start = q_row_ptrs[current_mapping_vertex];
            uint8_t parent_num = q_row_ptrs[current_mapping_vertex + 1] - parent_start;
            if (reuse_pos[current_mapping_vertex] != 0xFF && reuse_pos[current_mapping_vertex] >= init_layer)
            {
                uintV *current_set = reuse_store + reuse_pos[current_mapping_vertex] * layer_size;
                reuse_size[current_mapping_vertex] = reuse_size[reuse_pos[current_mapping_vertex]];
                bool flag = false;
                for (uint8_t parent = parent_start; parent < parent_start + parent_num; parent++)
                {
                    if (reuse_q_cols[parent])
                        continue;
                    flag = true;
                    uintV parent_vid = partial_match[q_cols[parent]];
                    BinaryIntersection(current_set, cols + row_ptrs[parent_vid], reuse_store_array, reuse_size[current_mapping_vertex], row_ptrs[parent_vid + 1] - row_ptrs[parent_vid]);
                    __syncwarp();
                    current_set = reuse_store_array;
                }
                if (!flag)
                {
                    for (uintE i = threadIdinWarp; i < reuse_size[current_mapping_vertex]; i++)
                        reuse_store_array[i] = current_set[i];
                }
            }
            else
            {
                uintV *current_set = cols + row_ptrs[partial_match[q_cols[parent_start]]];
                reuse_size[current_mapping_vertex] = row_ptrs[partial_match[q_cols[parent_start]] + 1] - row_ptrs[partial_match[q_cols[parent_start]]];
                if (parent_num == 1)
                {
                    for (uintE i = threadIdinWarp; i < reuse_size[current_mapping_vertex]; i++)
                        reuse_store_array[i] = current_set[i];
                }
                for (uint8_t parent = parent_start + 1; parent < parent_start + parent_num; parent++)
                {
                    uintV parent_vid = partial_match[q_cols[parent]];
                    BinaryIntersection(current_set, cols + row_ptrs[parent_vid], reuse_store_array, reuse_size[current_mapping_vertex], row_ptrs[parent_vid + 1] - row_ptrs[parent_vid]);
                    __syncwarp();
                    current_set = reuse_store_array;
                }
            }

            if (restriction[current_mapping_vertex] != 0xFF)
            {
                uintV restrict_vid = partial_match[restriction[current_mapping_vertex]];
                uintV write_pos = 0;
                for (uintE i = threadIdinWarp; i < reuse_size[current_mapping_vertex]; i += 32)
                {
                    bool flag = true;
                    uintV v = reuse_store_array[i];
                    if (v < restrict_vid)
                    {
                        flag = false;
                    }
                    if (flag)
                        for (uint8_t j = 0; j < current_mapping_vertex; j++)
                        {
                            if (v == partial_match[j])
                            {
                                flag = false;
                                break;
                            }
                        }
                    __syncwarp(__activemask());
                    if (flag)
                    {
                        coalesced_group active = cooperative_groups::coalesced_threads();
                        uintV wptr = write_pos + active.thread_rank();
                        stack_array[wptr] = v;
                    }
                    write_pos += __reduce_add_sync(__activemask(), flag);
                }
                write_pos = __shfl_sync(0xFFFFFFFF, write_pos, 0);
                __syncwarp();
                stack_size[current_mapping_vertex] = __shfl_sync(0xFFFFFFFF, write_pos, 0);
            }
            else
            {
                uintV write_pos = 0;
                for (uintE i = threadIdinWarp; i < reuse_size[current_mapping_vertex]; i += 32)
                {
                    bool flag = true;
                    uintV v = reuse_store_array[i];
                    for (uint8_t j = 0; j < current_mapping_vertex; j++)
                    {
                        if (v == partial_match[j])
                        {
                            flag = false;
                            break;
                        }
                    }
                    __syncwarp(__activemask());
                    if (flag)
                    {
                        coalesced_group active = cooperative_groups::coalesced_threads();
                        uintV wptr = write_pos + active.thread_rank();
                        stack_array[wptr] = v;
                    }
                    write_pos += __reduce_add_sync(__activemask(), flag);
                }
                stack_size[current_mapping_vertex] = __shfl_sync(0xFFFFFFFF, write_pos, 0);
            }
            if (current_mapping_vertex == q_vertex_count - 1 || stack_size[current_mapping_vertex] == 0)
            {
                if (threadIdinWarp == 0)
                {
                    warpCounter += stack_size[current_mapping_vertex];
                }
                __syncwarp();
                current_mapping_vertex--;
                while (current_mapping_vertex >= init_layer && stack_iter[current_mapping_vertex] == stack_size[current_mapping_vertex] - 1)
                    current_mapping_vertex--;
                if (current_mapping_vertex == init_layer - 1)
                    break;
                stack_iter[current_mapping_vertex]++;
                partial_match[current_mapping_vertex] =
                    stacks[(current_mapping_vertex)*layer_size +
                           stack_iter[current_mapping_vertex]];
                current_mapping_vertex++;
            }
            else
            {
                stack_iter[current_mapping_vertex] = 0;
                partial_match[current_mapping_vertex] =
                    stacks[(current_mapping_vertex)*layer_size];
                current_mapping_vertex++;
            }
        }
    }
    __syncthreads();
}

__launch_bounds__(1024, 1)
    __global__ void EdgeCentricReuseDFSKernel(
        uintV vertex_count,
        uintE *row_ptrs,
        uintV *cols,
        uint32_t layer_size,
        uintV *BFSlayer,
        uintV *stacks,
        uintV *reuse_store,
        uint8_t q_vertex_count,
        const uint32_t *__restrict__ q_row_ptrs,
        const uint8_t *__restrict__ q_cols,
        const uint8_t *__restrict__ restriction,
        const uint8_t *__restrict__ reuse_pos,
        const uint8_t *__restrict__ reuse_q_cols,
        uintE init_edge_num,
        uintV *init_edges,
        uint64_t *Counter)
{
    __shared__ uintV partial_match[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_iter[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_size[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t reuse_size[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    // __shared__ uintV shared_reuse_store[WARP_PER_BLOCK * MAX_REUSE_STORE];
    __shared__ uint64_t warpCounter[WARP_PER_BLOCK];
#ifdef PROFILE
    __shared__ uint64_t total_clock_start;
    __shared__ uint64_t total_clocks;
#endif
    __shared__ uint32_t BFS_layer_size;
    uintE init_match_pointer;
    uint8_t init_layer = 3;
    warpCounter[warpIdinBlock] = 0;
#ifdef PROFILE
    if (threadIdx.x == 0)
    {
        total_clocks = 0;
        total_clock_start = clock64();
    }
#endif
    for (init_match_pointer = blockIdx.x; init_match_pointer < init_edge_num; init_match_pointer += gridDim.x)
    {
        uint8_t current_mapping_vertex = 2;
        if (threadIdinWarp < current_mapping_vertex)
            partial_match[warpIdinBlock * MAX_QUERY_VERTEX + threadIdinWarp] = init_edges[init_match_pointer * 2 + threadIdinWarp];
        __syncthreads();
        extendWithReuse(
            vertex_count,
            row_ptrs,
            cols,
            layer_size,
            BFSlayer + blockIdx.x * layer_size,
            stacks + warpId * (q_vertex_count)*layer_size,
            reuse_store + warpId * (q_vertex_count)*layer_size,
            q_vertex_count,
            q_row_ptrs,
            q_cols,
            restriction,
            reuse_pos,
            reuse_q_cols,
            2,
            partial_match + warpIdinBlock * MAX_QUERY_VERTEX,
            stack_iter + warpIdinBlock * MAX_QUERY_VERTEX,
            stack_size + warpIdinBlock * MAX_QUERY_VERTEX,
            reuse_size + warpIdinBlock * MAX_QUERY_VERTEX,
            warpCounter[warpIdinBlock],
            BFS_layer_size,
            Counter);
        __syncthreads();
    }
#ifdef PROFILE
    if (threadIdx.x == 0)
        total_clocks += clock64() - total_clock_start;
#endif
#ifdef PROFILE
    if (threadIdx.x == 0)
        printf("blockId: %d, %lu\n", blockIdx.x, total_clocks);
#endif
    if (threadIdinWarp == 0)
    {
        atomicAdd((unsigned long long *)Counter, (unsigned long long)warpCounter[warpIdinBlock]);
    }
}

__forceinline__ __device__ void extendWithReuseOffload(
    uintV vertex_count,
    uintE *row_ptrs,
    uintV *cols,
    uint32_t layer_size,
    uintV *BFSlayer,
    uintV *stacks,
    uintV *reuse_store,
    uint8_t q_vertex_count,
    const uint32_t *__restrict__ q_row_ptrs,
    const uint8_t *__restrict__ q_cols,
    const uint8_t *__restrict__ restriction,
    const uint8_t *__restrict__ reuse_pos,
    const uint8_t *__restrict__ reuse_q_cols,
    uint8_t init_vertex_num,
    uintV *partial_match,
    uint32_t *stack_iter,
    uint32_t *stack_size,
    uint32_t *reuse_size,
    uint64_t &warpCounter,
    uint32_t &BFS_layer_size,
    uint64_t *Counter,
    uint8_t offload_layer,
    WorkerQueue *queue
    // #ifdef PROFILE
    //     ,uint64_t& balance_clock_start,
    //     uint64_t& balance_clocks
    // #endif
)
{
    uint8_t current_mapping_vertex = init_vertex_num;
    uint8_t parent_start = q_row_ptrs[current_mapping_vertex];
    uint8_t parent_num = q_row_ptrs[current_mapping_vertex + 1] - parent_start;
    uintV pivot_vertex = partial_match[q_cols[parent_start]];
    if (threadIdx.x == 0)
        BFS_layer_size = 0;
    __syncthreads();
    for (uintV i = row_ptrs[pivot_vertex] + threadIdx.x; i < row_ptrs[pivot_vertex + 1]; i += blockDim.x)
    {
        bool flag = true;
        uintV candidate = cols[i];
        if (restriction[current_mapping_vertex] != 0xFF && candidate <= partial_match[restriction[current_mapping_vertex]])
        {
            flag = false;
        }
        for (uint8_t j = 0; j < current_mapping_vertex; j++)
        {
            if (candidate == partial_match[j])
            {
                flag = false;
            }
        }
        if (parent_num > 1 && flag)
        {
            for (uint8_t j = 1; j < parent_num; j++)
            {
                uintV parent = partial_match[q_cols[parent_start + j]];
                if (!BinarySearch(candidate, cols + row_ptrs[parent],
                                  row_ptrs[parent + 1] - row_ptrs[parent]))
                {
                    flag = false;
                }
            }
        }
        __syncwarp(__activemask());
        uint32_t wptr_base;
        uint32_t candidate_num = __reduce_add_sync(__activemask(), flag);
        if (threadIdinWarp == 0)
            wptr_base = atomicAdd(&BFS_layer_size, candidate_num);
        wptr_base = __shfl_sync(__activemask(), wptr_base, 0);
        if (flag)
        {
            coalesced_group active = cooperative_groups::coalesced_threads();
            BFSlayer[wptr_base + active.thread_rank()] = candidate;
        }
        __syncwarp(__activemask());
    }
    __syncthreads();
    if (current_mapping_vertex == q_vertex_count - 1)
    {
        if (threadIdx.x == 0)
        {
            atomicAdd((unsigned long long *)Counter, BFS_layer_size);
        }
        __syncthreads();
        return;
    }
    uint8_t init_layer = init_vertex_num + 1;
    for (uint32_t subtask_id = warpIdinBlock; subtask_id < BFS_layer_size; subtask_id += WARP_PER_BLOCK)
    {
        if (threadIdinWarp == 0)
        {
            partial_match[current_mapping_vertex] = BFSlayer[subtask_id];
        }
        __syncwarp();
        current_mapping_vertex++;
        while (true)
        {
            if (current_mapping_vertex <= offload_layer && queue->queueCount)
            {
                if (inform(*queue, partial_match, current_mapping_vertex))
                {
                    __syncwarp();
                    current_mapping_vertex--;
                    while (current_mapping_vertex >= init_layer && stack_iter[current_mapping_vertex] == stack_size[current_mapping_vertex] - 1)
                        current_mapping_vertex--;
                    if (current_mapping_vertex == init_layer - 1)
                        break;
                    stack_iter[current_mapping_vertex]++;
                    partial_match[current_mapping_vertex] =
                        stacks[(current_mapping_vertex)*layer_size +
                               stack_iter[current_mapping_vertex]];
                    current_mapping_vertex++;
                }
            }
            __syncwarp();
            uintV *stack_array = stacks + (current_mapping_vertex)*layer_size;
            uintV *reuse_store_array = reuse_store + (current_mapping_vertex)*layer_size;
            uint8_t parent_start = q_row_ptrs[current_mapping_vertex];
            uint8_t parent_num = q_row_ptrs[current_mapping_vertex + 1] - parent_start;
            if (reuse_pos[current_mapping_vertex] != 0xFF && reuse_pos[current_mapping_vertex] >= init_layer)
            {
                uintV *current_set = reuse_store + reuse_pos[current_mapping_vertex] * layer_size;
                reuse_size[current_mapping_vertex] = reuse_size[reuse_pos[current_mapping_vertex]];
                bool flag = false;
                for (uint8_t parent = parent_start; parent < parent_start + parent_num; parent++)
                {
                    if (reuse_q_cols[parent])
                        continue;
                    flag = true;
                    uintV parent_vid = partial_match[q_cols[parent]];
                    BinaryIntersection(current_set, cols + row_ptrs[parent_vid], reuse_store_array, reuse_size[current_mapping_vertex], row_ptrs[parent_vid + 1] - row_ptrs[parent_vid]);
                    __syncwarp();
                    current_set = reuse_store_array;
                }
                if (!flag) {
                    for (uintE i = threadIdinWarp; i < reuse_size[current_mapping_vertex]; i++)
                        reuse_store_array[i] = current_set[i];
                }
            }
            else
            {
                uintV *current_set = cols + row_ptrs[partial_match[q_cols[parent_start]]];
                reuse_size[current_mapping_vertex] = row_ptrs[partial_match[q_cols[parent_start]] + 1] - row_ptrs[partial_match[q_cols[parent_start]]];
                if (parent_num == 1)
                {
                    for (uintE i = threadIdinWarp; i < reuse_size[current_mapping_vertex]; i++)
                        reuse_store_array[i] = current_set[i];
                }
                for (uint8_t parent = parent_start + 1; parent < parent_start + parent_num; parent++)
                {
                    uintV parent_vid = partial_match[q_cols[parent]];
                    BinaryIntersection(current_set, cols + row_ptrs[parent_vid], reuse_store_array, reuse_size[current_mapping_vertex], row_ptrs[parent_vid + 1] - row_ptrs[parent_vid]);
                    __syncwarp();
                    current_set = reuse_store_array;
                }
            }

            if (restriction[current_mapping_vertex] != 0xFF)
            {
                uintV restrict_vid = partial_match[restriction[current_mapping_vertex]];
                uintV write_pos = 0;
                for (uintE i = threadIdinWarp; i < reuse_size[current_mapping_vertex]; i += 32)
                {
                    bool flag = true;
                    uintV v = reuse_store_array[i];
                    if (v < restrict_vid)
                    {
                        flag = false;
                    }
                    if (flag)
                        for (uint8_t j = 0; j < current_mapping_vertex; j++)
                        {
                            if (v == partial_match[j])
                            {
                                flag = false;
                                break;
                            }
                        }
                    __syncwarp(__activemask());
                    if (flag)
                    {
                        coalesced_group active = cooperative_groups::coalesced_threads();
                        uintV wptr = write_pos + active.thread_rank();
                        stack_array[wptr] = v;
                    }
                    write_pos += __reduce_add_sync(__activemask(), flag);
                }
                write_pos = __shfl_sync(0xFFFFFFFF, write_pos, 0);
                __syncwarp();
                stack_size[current_mapping_vertex] = __shfl_sync(0xFFFFFFFF, write_pos, 0);
            }
            else
            {
                uintV write_pos = 0;
                for (uintE i = threadIdinWarp; i < reuse_size[current_mapping_vertex]; i += 32)
                {
                    bool flag = true;
                    uintV v = reuse_store_array[i];
                    for (uint8_t j = 0; j < current_mapping_vertex; j++)
                    {
                        if (v == partial_match[j])
                        {
                            flag = false;
                            break;
                        }
                    }
                    __syncwarp(__activemask());
                    if (flag)
                    {
                        coalesced_group active = cooperative_groups::coalesced_threads();
                        uintV wptr = write_pos + active.thread_rank();
                        stack_array[wptr] = v;
                    }
                    write_pos += __reduce_add_sync(__activemask(), flag);
                }
                stack_size[current_mapping_vertex] = __shfl_sync(0xFFFFFFFF, write_pos, 0);
            }
            if (current_mapping_vertex == q_vertex_count - 1 || stack_size[current_mapping_vertex] == 0)
            {
                if (threadIdinWarp == 0)
                {
                    warpCounter += stack_size[current_mapping_vertex];
                }
                __syncwarp();
                current_mapping_vertex--;
                while (current_mapping_vertex >= init_layer && stack_iter[current_mapping_vertex] == stack_size[current_mapping_vertex] - 1)
                    current_mapping_vertex--;
                if (current_mapping_vertex == init_layer - 1)
                    break;
                stack_iter[current_mapping_vertex]++;
                partial_match[current_mapping_vertex] =
                    stacks[(current_mapping_vertex)*layer_size +
                           stack_iter[current_mapping_vertex]];
                current_mapping_vertex++;
            }
            else
            {
                stack_iter[current_mapping_vertex] = 0;
                partial_match[current_mapping_vertex] =
                    stacks[(current_mapping_vertex)*layer_size];
                current_mapping_vertex++;
            }
        }
    }
    __syncthreads();
}

__launch_bounds__(1024, 1)
    __global__ void EdgeCentricBalancedReuseDFSKernel(
        uintV vertex_count,
        uintE *row_ptrs,
        uintV *cols,
        uint32_t layer_size,
        uintV *BFSlayer,
        uintV *stacks,
        uintV *reuse_store,
        uint8_t q_vertex_count,
        const uint32_t *__restrict__ q_row_ptrs,
        const uint8_t *__restrict__ q_cols,
        const uint8_t *__restrict__ restriction,
        const uint8_t *__restrict__ reuse_pos,
        const uint8_t *__restrict__ reuse_q_cols,
        uintE init_edge_num,
        uintV *init_edges,
        uint64_t *Counter,
        uint8_t offload_layer,
        WorkerQueue *queue)
{
    __shared__ uintV partial_match[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_iter[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_size[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t reuse_size[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint64_t warpCounter[WARP_PER_BLOCK];
#ifdef PROFILE
    __shared__ uint64_t total_clock_start;
    __shared__ uint64_t total_clocks;
    __shared__ uint64_t traverse_clock_start;
    __shared__ uint64_t traverse_clocks;
    __shared__ uint64_t wait_clock_start;
    __shared__ uint64_t wait_clocks;
#endif
    __shared__ uint32_t queuepos;
    __shared__ uint32_t BFS_layer_size;
    uintE init_match_pointer;
    uint8_t init_layer = 3;
    warpCounter[warpIdinBlock] = 0;
#ifdef PROFILE
    if (threadIdx.x == 0)
    {
        total_clocks = 0;
        traverse_clocks = 0;
        wait_clocks = 0;
        total_clock_start = clock64();
    }
#endif
    for (init_match_pointer = blockIdx.x; init_match_pointer < init_edge_num; init_match_pointer += gridDim.x)
    {
        uint8_t current_mapping_vertex = 2;
        if (threadIdinWarp < current_mapping_vertex)
            partial_match[warpIdinBlock * MAX_QUERY_VERTEX + threadIdinWarp] = init_edges[init_match_pointer * 2 + threadIdinWarp];
        __syncthreads();

#ifdef PROFILE
        if (threadIdx.x == 0)
        {
            traverse_clock_start = clock64();
        }
#endif
        extendWithReuseOffload(
            vertex_count,
            row_ptrs,
            cols,
            layer_size,
            BFSlayer + blockIdx.x * layer_size,
            stacks + warpId * (q_vertex_count)*layer_size,
            reuse_store + warpId * (q_vertex_count)*layer_size,
            q_vertex_count,
            q_row_ptrs,
            q_cols,
            restriction,
            reuse_pos,
            reuse_q_cols,
            2,
            partial_match + warpIdinBlock * MAX_QUERY_VERTEX,
            stack_iter + warpIdinBlock * MAX_QUERY_VERTEX,
            stack_size + warpIdinBlock * MAX_QUERY_VERTEX,
            reuse_size + warpIdinBlock * MAX_QUERY_VERTEX,
            warpCounter[warpIdinBlock],
            BFS_layer_size,
            Counter,
            offload_layer,
            queue
            // #ifdef PROFILE
            //             ,balance_clock_start,
            //             balance_clocks
            // #endif
        );
        __syncthreads();

#ifdef PROFILE
        if (threadIdx.x == 0)
        {
            traverse_clocks += clock64() - traverse_clock_start;
        }
#endif
    }
    __syncthreads();

    while (true)
    {
#ifdef PROFILE
        if (threadIdx.x == 0)
            wait_clock_start = clock64();
#endif
        enqueue(*queue, queuepos);
        uint32_t init_vertex_num;
        if (waitTask(*queue, queuepos, partial_match + warpIdinBlock * MAX_QUERY_VERTEX, init_vertex_num))
        {

            dequeue(*queue, queuepos);

#ifdef PROFILE
            if (threadIdx.x == 0)
            {
                wait_clocks += clock64() - wait_clock_start;
                traverse_clock_start = clock64();
            }
#endif
            extendWithReuseOffload(
                vertex_count,
                row_ptrs,
                cols,
                layer_size,
                BFSlayer + blockIdx.x * layer_size,
                stacks + warpId * (q_vertex_count)*layer_size,
                reuse_store + warpId * (q_vertex_count)*layer_size,
                q_vertex_count,
                q_row_ptrs,
                q_cols,
                restriction,
                reuse_pos,
                reuse_q_cols,
                init_vertex_num,
                partial_match + warpIdinBlock * MAX_QUERY_VERTEX,
                stack_iter + warpIdinBlock * MAX_QUERY_VERTEX,
                stack_size + warpIdinBlock * MAX_QUERY_VERTEX,
                reuse_size + warpIdinBlock * MAX_QUERY_VERTEX,
                warpCounter[warpIdinBlock],
                BFS_layer_size,
                Counter,
                offload_layer,
                queue);

#ifdef PROFILE
            if (threadIdx.x == 0)
            {
                traverse_clocks += clock64() - traverse_clock_start;
            }
#endif
        }
        else
        {

#ifdef PROFILE
            if (threadIdx.x == 0)
                wait_clocks += clock64() - wait_clock_start;
#endif
            break;
        }
    }
#ifdef PROFILE
    if (threadIdx.x == 0)
        total_clocks += clock64() - total_clock_start;
#endif
#ifdef PROFILE
    if (threadIdx.x == 0)
        printf("blockId: %d, %lu, %lu, %lu\n", blockIdx.x, total_clocks, wait_clocks, traverse_clocks);
#endif
    if (threadIdinWarp == 0)
    {
        atomicAdd((unsigned long long *)Counter, (unsigned long long)warpCounter[warpIdinBlock]);
    }
}