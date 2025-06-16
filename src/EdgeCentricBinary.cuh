#pragma once
#include "Meta.h"
#include <cuda.h>
#include "utils/GPUUtil.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "Balance.cuh"
#include "Matchutil.cuh"
using namespace cooperative_groups;

__forceinline__ __device__ void extendOffloadBinaryDFS(
    uintV *vertex_list,
    uint32_t layer_size,
    uint32_t indSize,
    uint32_t *stacks,
    uint32_t *InducedGraph,
    uint32_t *Vorient,
    uint32_t *Vind,
    uint8_t q_vertex_count,
    const uint32_t *__restrict__ q_row_ptrs,
    const uint8_t *__restrict__ q_cols,
    const uint8_t *__restrict__ restriction,
    uint8_t init_vertex_num,
    uintV *partial_match,
    uint32_t *stack_iter,
    uint32_t *stack_size,
    uint64_t &warpCounter,
    uint8_t offload_layer,
    WorkerQueue *queue)
{
    uint8_t current_mapping_vertex = init_vertex_num;
    while (true)
    {
        if (current_mapping_vertex <= offload_layer && queue->queueCount)
        {
            if (inform(*queue, partial_match, current_mapping_vertex))
            {
                Vind[stack_iter[current_mapping_vertex] / 32] |= (1 << stack_iter[current_mapping_vertex] % 32);
                current_mapping_vertex--;
                while (current_mapping_vertex >= init_vertex_num)
                {
                    Vind[stack_iter[current_mapping_vertex] / 32] |= (1 << stack_iter[current_mapping_vertex] % 32);
                    FindFirstOneBinary(stacks + current_mapping_vertex * layer_size,
                                       indSize, stack_iter[current_mapping_vertex] + 1, stack_iter[current_mapping_vertex]);
                    if (stack_iter[current_mapping_vertex] == 0xFFFFFFFFU)
                    {
                        current_mapping_vertex--;
                    }
                    else
                    {
                        Vind[stack_iter[current_mapping_vertex] / 32] &= ~(1U << (stack_iter[current_mapping_vertex] % 32));
                        partial_match[current_mapping_vertex] = vertex_list[stack_iter[current_mapping_vertex]];
                        break;
                    }
                }
                if (current_mapping_vertex == init_vertex_num - 1)
                {
                    break;
                }
                current_mapping_vertex++;
            }
        }
        __syncwarp();
        uint32_t *stack_array = stacks + current_mapping_vertex * layer_size;
        uint8_t parent_start = q_row_ptrs[current_mapping_vertex];
        uint8_t parent_num = q_row_ptrs[current_mapping_vertex + 1] - parent_start;
        for (uint32_t i = threadIdinWarp; i < (indSize + 31) / 32; i += 32)
        {
            stack_array[i] = Vind[i];
        }
        if (restriction[current_mapping_vertex] != 0xFF)
        {
            if (restriction[current_mapping_vertex] == 0)
            {
                SetExclusiveBinary(stack_array, Vorient, indSize);
            }
            else
            {
                for (uint32_t j = threadIdinWarp; j < stack_iter[restriction[current_mapping_vertex]] / 32; j += 32)
                {
                    stack_array[j] = 0;
                }
                if (stack_iter[restriction[current_mapping_vertex]] % 32)
                {
                    stack_array[stack_iter[restriction[current_mapping_vertex]] / 32] &=
                        (0xFFFFFFFFU << (stack_iter[restriction[current_mapping_vertex]] % 32));
                }
                __syncwarp();
            }
        }
        for (uint8_t parent = 0; parent < parent_num; parent++)
        {
            if (q_cols[parent + parent_start] == 0)
                continue;
            SetIntersectionBinary(stack_array, InducedGraph + stack_iter[q_cols[parent + parent_start]] * layer_size, indSize);
        }
        CountOnesBinary(stack_array, indSize, stack_size[current_mapping_vertex]);
        if (current_mapping_vertex == q_vertex_count - 1 || stack_size[current_mapping_vertex] == 0)
        {
            if (threadIdinWarp == 0)
            {
                warpCounter += stack_size[current_mapping_vertex];
            }
            __syncwarp();
            current_mapping_vertex--;
            while (current_mapping_vertex >= init_vertex_num)
            {
                Vind[stack_iter[current_mapping_vertex] / 32] |= (1 << stack_iter[current_mapping_vertex] % 32);
                FindFirstOneBinary(stacks + current_mapping_vertex * layer_size,
                                   indSize, stack_iter[current_mapping_vertex] + 1, stack_iter[current_mapping_vertex]);
                if (stack_iter[current_mapping_vertex] == 0xFFFFFFFFU)
                {
                    current_mapping_vertex--;
                }
                else
                {
                    Vind[stack_iter[current_mapping_vertex] / 32] &= ~(1U << (stack_iter[current_mapping_vertex] % 32));
                    partial_match[current_mapping_vertex] = vertex_list[stack_iter[current_mapping_vertex]];
                    break;
                }
            }
            if (current_mapping_vertex == init_vertex_num - 1)
                break;
            current_mapping_vertex++;
        }
        else
        {
            FindFirstOneBinary(stack_array, indSize, 0, stack_iter[current_mapping_vertex]);
            Vind[stack_iter[current_mapping_vertex] / 32] &= ~(1U << (stack_iter[current_mapping_vertex] % 32));
            partial_match[current_mapping_vertex] = vertex_list[stack_iter[current_mapping_vertex]];
            current_mapping_vertex++;
        }
    }
}
__forceinline__ __device__ void extendOffloadBinary(
    uintV vertex_count,
    uintE *row_ptrs,
    uintV *cols,
    uint32_t layer_size,
    uintV *BFSlayer,
    uint32_t *stacks,
    uint32_t *InducedGraph,
    uint32_t *Vorient,
    uint8_t q_vertex_count,
    const uint32_t *__restrict__ q_row_ptrs,
    const uint8_t *__restrict__ q_cols,
    const uint8_t *__restrict__ restriction,
    uint8_t current_mapping_vertex,
    uintV *partial_match,
    uint32_t *stack_iter,
    uint32_t *stack_size,
    uint64_t &warpCounter,
    uint32_t &BFS_layer_size,
    uint64_t *Counter,
    uint8_t offload_layer,
    WorkerQueue *queue
#ifdef PROFILE
    ,
    uint64_t &induce_clock_start,
    uint64_t &induce_clocks
#endif
)
{
    uint32_t indSize;
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
#ifdef PROFILE
    if (threadIdx.x == 0)
        induce_clock_start = clock64();
#endif
    ComputeInducedGraph(vertex_count, row_ptrs, cols, layer_size / 32, InducedGraph, Vorient, partial_match[0], indSize);
#ifdef PROFILE
    if (threadIdx.x == 0)
        induce_clocks = clock64() - induce_clock_start;
#endif
    for (uint32_t subtask_id = warpIdinBlock; subtask_id < BFS_layer_size; subtask_id += WARP_PER_BLOCK)
    {
        if (threadIdinWarp == 0)
        {
            partial_match[current_mapping_vertex] = BFSlayer[subtask_id];
        }
        for (int i = threadIdinWarp; i < (indSize + 31) / 32; i += 32)
        {
            stacks[i] = 0xFFFFFFFFU >> (((i + 1) * 32 <= indSize) ? 0 : ((i + 1) * 32 - indSize));
        }
        __syncwarp();
        if (threadIdinWarp <= current_mapping_vertex && threadIdinWarp != 0)
        {
            uint32_t pos = BinarySearchPosition(partial_match[threadIdinWarp], cols + row_ptrs[partial_match[0]], indSize);
            stacks[threadIdinWarp * layer_size / 32 + pos / 32] |= (1 << (pos % 32));
            atomicAnd(stacks + pos / 32, ~(1 << (pos % 32)));
            stack_iter[threadIdinWarp] = pos;
            stack_size[threadIdinWarp] = 1;
        }
        __syncwarp();
        extendOffloadBinaryDFS(
            cols + row_ptrs[partial_match[0]],
            layer_size / 32,
            indSize,
            stacks,
            InducedGraph,
            Vorient,
            stacks,
            q_vertex_count,
            q_row_ptrs,
            q_cols,
            restriction,
            current_mapping_vertex + 1,
            partial_match,
            stack_iter,
            stack_size,
            warpCounter,
            offload_layer,
            queue);
    }
}

__launch_bounds__(1024, 1)
    __global__ void EdgeCentricDFSBinaryBalancedKernel(
        uintV vertex_count,
        uintE *row_ptrs,
        uintV *cols,
        uint32_t layer_size,
        uint32_t max_degree,
        uintV *BFSlayer,
        uint32_t *stacks,
        uint32_t *InducedGraph,
        uint32_t *Vorient,
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
    __shared__ uint64_t induce_clock_start;
    __shared__ uint64_t induce_clocks;
    __shared__ uint64_t wait_clock_start;
    __shared__ uint64_t wait_clocks;
#endif
    __shared__ uint32_t queuepos;
    __shared__ uint32_t BFS_layer_size;

    uintE init_match_pointer;
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
        if (threadIdinWarp < 2)
            partial_match[warpIdinBlock * MAX_QUERY_VERTEX + threadIdinWarp] = init_edges[init_match_pointer * 2 + threadIdinWarp];
        __syncthreads();
#ifdef PROFILE
        if (threadIdx.x == 0)
        {
            traverse_clock_start = clock64();
        }
#endif
        extendOffloadBinary(
            vertex_count,
            row_ptrs,
            cols,
            layer_size,
            BFSlayer + blockIdx.x * layer_size,
            stacks + warpId * q_vertex_count * layer_size / 32,
            InducedGraph + blockIdx.x * max_degree * layer_size / 32,
            Vorient + blockIdx.x * layer_size / 32,
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
            queue
#ifdef PROFILE
            ,induce_clock_start,
            induce_clocks
#endif
        );
        __syncthreads();
#ifdef PROFILE
        if (threadIdx.x == 0)
        {
            traverse_clocks += clock64() - traverse_clock_start;
        }
#endif
    }
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
            assert(init_vertex_num == 3);
            extendOffloadBinary(
                vertex_count,
                row_ptrs,
                cols,
                layer_size,
                BFSlayer + blockIdx.x * layer_size,
                stacks + warpId * q_vertex_count * layer_size / 32,
                InducedGraph + blockIdx.x * max_degree * layer_size / 32,
                Vorient + blockIdx.x * layer_size / 32,
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
                queue
#ifdef PROFILE
                ,induce_clock_start,
                induce_clocks
#endif
            );
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
        printf("blockId: %d, %lu, %lu, %lu, %lu\n", blockIdx.x, total_clocks, wait_clocks, traverse_clocks, induce_clocks);
#endif
    if (threadIdinWarp == 0)
    {
        atomicAdd((unsigned long long *)Counter, (unsigned long long)warpCounter[warpIdinBlock]);
    }
}

__forceinline__ __device__ void extendBinaryDFS(
    uintV *vertex_list,
    uint32_t layer_size,
    uint32_t indSize,
    uint32_t *stacks,
    uint32_t *InducedGraph,
    uint32_t *Vorient,
    uint32_t *Vind,
    uint8_t q_vertex_count,
    const uint32_t *__restrict__ q_row_ptrs,
    const uint8_t *__restrict__ q_cols,
    const uint8_t *__restrict__ restriction,
    uint8_t init_vertex_num,
    uintV *partial_match,
    uint32_t *stack_iter,
    uint32_t *stack_size,
    uint64_t &warpCounter)
{
    uint8_t current_mapping_vertex = init_vertex_num;
    while (true)
    {
        __syncwarp();
        uint32_t *stack_array = stacks + current_mapping_vertex * layer_size;
        uint8_t parent_start = q_row_ptrs[current_mapping_vertex];
        uint8_t parent_num = q_row_ptrs[current_mapping_vertex + 1] - parent_start;
        for (uint32_t i = threadIdinWarp; i < (indSize + 31) / 32; i += 32)
        {
            stack_array[i] = Vind[i];
        }
        if (restriction[current_mapping_vertex] != 0xFF)
        {
            if (restriction[current_mapping_vertex] == 0)
            {
                SetExclusiveBinary(stack_array, Vorient, indSize);
            }
            else
            {
                for (uint32_t j = threadIdinWarp; j < stack_iter[restriction[current_mapping_vertex]] / 32; j += 32)
                {
                    stack_array[j] = 0;
                }
                if (stack_iter[restriction[current_mapping_vertex]] % 32)
                {
                    stack_array[stack_iter[restriction[current_mapping_vertex]] / 32] &=
                        (0xFFFFFFFFU << (stack_iter[restriction[current_mapping_vertex]] % 32));
                }
                __syncwarp();
            }
        }
        for (uint8_t parent = 0; parent < parent_num; parent++)
        {
            if (q_cols[parent + parent_start] == 0)
                continue;
            SetIntersectionBinary(stack_array, InducedGraph + stack_iter[q_cols[parent + parent_start]] * layer_size, indSize);
        }
        CountOnesBinary(stack_array, indSize, stack_size[current_mapping_vertex]);
        if (current_mapping_vertex == q_vertex_count - 1 || stack_size[current_mapping_vertex] == 0)
        {
            if (threadIdinWarp == 0)
            {
                warpCounter += stack_size[current_mapping_vertex];
            }
            __syncwarp();
            current_mapping_vertex--;
            while (current_mapping_vertex >= init_vertex_num)
            {
                Vind[stack_iter[current_mapping_vertex] / 32] |= (1 << stack_iter[current_mapping_vertex] % 32);
                FindFirstOneBinary(stacks + current_mapping_vertex * layer_size,
                                   indSize, stack_iter[current_mapping_vertex] + 1, stack_iter[current_mapping_vertex]);
                if (stack_iter[current_mapping_vertex] == 0xFFFFFFFFU)
                {
                    current_mapping_vertex--;
                }
                else
                {
                    Vind[stack_iter[current_mapping_vertex] / 32] &= ~(1U << (stack_iter[current_mapping_vertex] % 32));
                    partial_match[current_mapping_vertex] = vertex_list[stack_iter[current_mapping_vertex]];
                    break;
                }
            }
            if (current_mapping_vertex == init_vertex_num - 1)
                break;
            current_mapping_vertex++;
        }
        else
        {
            FindFirstOneBinary(stack_array, indSize, 0, stack_iter[current_mapping_vertex]);
            Vind[stack_iter[current_mapping_vertex] / 32] &= ~(1U << (stack_iter[current_mapping_vertex] % 32));
            partial_match[current_mapping_vertex] = vertex_list[stack_iter[current_mapping_vertex]];
            current_mapping_vertex++;
        }
    }
}

__forceinline__ __device__ void extendBinary(
    uintV vertex_count,
    uintE *row_ptrs,
    uintV *cols,
    uint32_t layer_size,
    uintV *BFSlayer,
    uint32_t *stacks,
    uint32_t *InducedGraph,
    uint32_t *Vorient,
    uint8_t q_vertex_count,
    const uint32_t *__restrict__ q_row_ptrs,
    const uint8_t *__restrict__ q_cols,
    const uint8_t *__restrict__ restriction,
    uint8_t current_mapping_vertex,
    uintV *partial_match,
    uint32_t *stack_iter,
    uint32_t *stack_size,
    uint64_t &warpCounter,
    uint32_t &BFS_layer_size,
    uint64_t *Counter
#ifdef PROFILE
    ,
    uint64_t &induce_clock_start,
    uint64_t &induce_clocks
#endif
)
{
    uint32_t indSize;
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
#ifdef PROFILE
    if (threadIdx.x == 0)
        induce_clock_start = clock64();
#endif
    ComputeInducedGraph(vertex_count, row_ptrs, cols, layer_size / 32, InducedGraph, Vorient, partial_match[0], indSize);
#ifdef PROFILE
    if (threadIdx.x == 0)
        induce_clocks = clock64() - induce_clock_start;
#endif
    for (uint32_t subtask_id = warpIdinBlock; subtask_id < BFS_layer_size; subtask_id += WARP_PER_BLOCK)
    {
        if (threadIdinWarp == 0)
        {
            partial_match[current_mapping_vertex] = BFSlayer[subtask_id];
        }
        for (int i = threadIdinWarp; i < (indSize + 31) / 32; i += 32)
        {
            stacks[i] = 0xFFFFFFFFU >> (((i + 1) * 32 <= indSize) ? 0 : ((i + 1) * 32 - indSize));
        }
        __syncwarp();
        if (threadIdinWarp <= current_mapping_vertex && threadIdinWarp != 0)
        {
            uint32_t pos = BinarySearchPosition(partial_match[threadIdinWarp], cols + row_ptrs[partial_match[0]], indSize);
            stacks[threadIdinWarp * layer_size / 32 + pos / 32] |= (1 << (pos % 32));
            atomicAnd(stacks + pos / 32, ~(1 << (pos % 32)));
            stack_iter[threadIdinWarp] = pos;
            stack_size[threadIdinWarp] = 1;
        }
        __syncwarp();
        extendBinaryDFS(
            cols + row_ptrs[partial_match[0]],
            layer_size / 32,
            indSize,
            stacks,
            InducedGraph,
            Vorient,
            stacks,
            q_vertex_count,
            q_row_ptrs,
            q_cols,
            restriction,
            current_mapping_vertex + 1,
            partial_match,
            stack_iter,
            stack_size,
            warpCounter);
    }
}

__launch_bounds__(1024, 1)
    __global__ void EdgeCentricDFSBinaryKernel(
        uintV vertex_count,
        uintE *row_ptrs,
        uintV *cols,
        uint32_t layer_size,
        uint32_t max_degree,
        uintV *BFSlayer,
        uint32_t *stacks,
        uint32_t *InducedGraph,
        uint32_t *Vorient,
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
    __shared__ uint64_t induce_clock_start;
    __shared__ uint64_t induce_clocks;
#endif
    __shared__ uint32_t queuepos;
    __shared__ uint32_t BFS_layer_size;

    uintE init_match_pointer;
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
        if (threadIdinWarp < 2)
            partial_match[warpIdinBlock * MAX_QUERY_VERTEX + threadIdinWarp] = init_edges[init_match_pointer * 2 + threadIdinWarp];
        __syncthreads();
        extendBinary(
            vertex_count,
            row_ptrs,
            cols,
            layer_size,
            BFSlayer + blockIdx.x * layer_size,
            stacks + warpId * q_vertex_count * layer_size / 32,
            InducedGraph + blockIdx.x * max_degree * layer_size / 32,
            Vorient + blockIdx.x * layer_size / 32,
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
            Counter
#ifdef PROFILE
            ,
            induce_clock_start,
            induce_clocks
#endif
        );
        __syncthreads();
    }
#ifdef PROFILE
    if (threadIdx.x == 0)
        total_clocks += clock64() - total_clock_start;
#endif
#ifdef PROFILE
    if (threadIdx.x == 0)
        printf("blockId: %d, %lu, %lu\n", blockIdx.x, total_clocks, induce_clocks);
#endif
    if (threadIdinWarp == 0)
    {
        atomicAdd((unsigned long long *)Counter, (unsigned long long)warpCounter[warpIdinBlock]);
    }
}
