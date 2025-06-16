#pragma once
#include "Meta.h"
#include <cuda.h>
#include "utils/GPUUtil.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "Balance.cuh"
#include "Matchutil.cuh"
using namespace cooperative_groups;

__launch_bounds__(1024, 1)
    __global__ void DFSKernel(
        uintV vertex_count,
        uintE *row_ptrs,
        uintV *cols,
        uint32_t layer_size,
        uintV *stacks,
        uint8_t q_vertex_count,
        const uint32_t *__restrict__ q_row_ptrs,
        const uint8_t *__restrict__ q_cols,
        const uint8_t *__restrict__ restriction,
        uintE init_block_tasks,
        uint8_t init_layer,
        uintE *match_ptrs,
        uintV *init_match,
        uintE *global_allocator,
        uint64_t *Counter)
{
    __shared__ uintV partial_match[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_iter[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_size[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint64_t warpCounter[WARP_PER_BLOCK];
    // __shared__ uintE init_match_pointer;
    uintE init_match_pointer;
    warpCounter[warpIdinBlock] = 0;
    // while(true)
    for (init_match_pointer = blockIdx.x; init_match_pointer < init_block_tasks; init_match_pointer += gridDim.x)
    {
        // __syncthreads();
        // if (threadIdx.x == 0)
        // {
        //     init_match_pointer = atomicAdd((unsigned long long*)global_allocator, 1UL);
        // }
        // __syncthreads();
        // if (init_match_pointer >= init_block_tasks) break;
        // block_match_allocator = match_ptrs[init_match_pointer];
        for (uintE match_id = match_ptrs[init_match_pointer] + warpIdinBlock; match_id < match_ptrs[init_match_pointer + 1]; match_id += WARP_PER_BLOCK)
        {
            if (threadIdinWarp < init_layer)
                partial_match[warpIdinBlock * MAX_QUERY_VERTEX + threadIdinWarp] = init_match[match_id * init_layer + threadIdinWarp];
            uint8_t current_mapping_vertex = init_layer;
            __syncwarp();
            while (true)
            {
                __syncwarp();
                uintV *stack_array = stacks + warpId * (q_vertex_count - init_layer) * layer_size + (current_mapping_vertex - init_layer) * layer_size;
                uint8_t parent_start = q_row_ptrs[current_mapping_vertex];
                uint8_t parent_num = q_row_ptrs[current_mapping_vertex + 1] - parent_start;
                if (restriction[current_mapping_vertex] != 0xFF)
                {
                    uintV restrict_vid = partial_match[warpIdinBlock * MAX_QUERY_VERTEX + restriction[current_mapping_vertex]];
                    uintV pivot_vid = partial_match[warpIdinBlock * MAX_QUERY_VERTEX + q_cols[parent_start]];
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
                                if (cols[i] == partial_match[warpIdinBlock * MAX_QUERY_VERTEX + j])
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
                    stack_size[warpIdinBlock * MAX_QUERY_VERTEX + current_mapping_vertex - init_layer] = __shfl_sync(0xFFFFFFFF, write_pos, 0);
                }
                else
                {
                    uintV pivot_vid = partial_match[warpIdinBlock * MAX_QUERY_VERTEX + q_cols[parent_start]];
                    uintV write_pos = 0;
                    for (uintE i = row_ptrs[pivot_vid] + threadIdinWarp; i < row_ptrs[pivot_vid + 1]; i += 32)
                    {
                        bool flag = true;
                        for (uint8_t j = 0; j < current_mapping_vertex; j++)
                        {
                            if (cols[i] == partial_match[warpIdinBlock * MAX_QUERY_VERTEX + j])
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
                    stack_size[warpIdinBlock * MAX_QUERY_VERTEX + current_mapping_vertex - init_layer] = __shfl_sync(0xFFFFFFFF, write_pos, 0);
                }
                for (uint8_t parent = parent_start + 1; parent < parent_start + parent_num; parent++)
                {
                    uintV parent_vid = partial_match[warpIdinBlock * MAX_QUERY_VERTEX + q_cols[parent]];
                    uintV before = stack_size[warpIdinBlock * MAX_QUERY_VERTEX + current_mapping_vertex - init_layer];
                    BinaryIntersection(stack_array, cols + row_ptrs[parent_vid], stack_size[warpIdinBlock * MAX_QUERY_VERTEX + current_mapping_vertex - init_layer], row_ptrs[parent_vid + 1] - row_ptrs[parent_vid]);
                    __syncwarp();
                }
                if (current_mapping_vertex == q_vertex_count - 1 || stack_size[warpIdinBlock * MAX_QUERY_VERTEX + current_mapping_vertex - init_layer] == 0)
                {
                    if (threadIdinWarp == 0)
                    {
                        warpCounter[warpIdinBlock] += stack_size[warpIdinBlock * MAX_QUERY_VERTEX + current_mapping_vertex - init_layer];
                    }
                    __syncwarp();
                    current_mapping_vertex--;
                    while (current_mapping_vertex >= init_layer && stack_iter[warpIdinBlock * MAX_QUERY_VERTEX + current_mapping_vertex - init_layer] == stack_size[warpIdinBlock * MAX_QUERY_VERTEX + current_mapping_vertex - init_layer] - 1)
                        current_mapping_vertex--;
                    if (current_mapping_vertex == init_layer - 1)
                        break;
                    stack_iter[warpIdinBlock * MAX_QUERY_VERTEX + current_mapping_vertex - init_layer]++;
                    partial_match[warpIdinBlock * MAX_QUERY_VERTEX + current_mapping_vertex] =
                        stacks[warpId * (q_vertex_count - init_layer) * layer_size + (current_mapping_vertex - init_layer) * layer_size +
                               stack_iter[warpIdinBlock * MAX_QUERY_VERTEX + current_mapping_vertex - init_layer]];
                    current_mapping_vertex++;
                }
                else
                {
                    stack_iter[warpIdinBlock * MAX_QUERY_VERTEX + current_mapping_vertex - init_layer] = 0;
                    partial_match[warpIdinBlock * MAX_QUERY_VERTEX + current_mapping_vertex] =
                        stacks[warpId * (q_vertex_count - init_layer) * layer_size + (current_mapping_vertex - init_layer) * layer_size];
                    current_mapping_vertex++;
                }
            }
        }
    }
    if (threadIdinWarp == 0)
    {
        atomicAdd((unsigned long long *)Counter, (unsigned long long)warpCounter[warpIdinBlock]);
    }
}


