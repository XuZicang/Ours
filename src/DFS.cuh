#pragma once
#include "Meta.h"
#include <cuda.h>
#include "utils/GPUUtil.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "Balance.cuh"
#include "TaskQueue.cuh"
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


struct MatchInfo {
    uintE* row_ptrs;
    uintV* cols;
    uintV* stacks;
    uint8_t q_row_ptrs[16];
    uint8_t q_cols[48];
    uint8_t q_restriction[16];
    uintV* init_tasks;
    uintE init_task_num;
    uintV dg_vertex_count;
    uint32_t stack_layer_size;
    uint8_t q_vertex_count;
    uint8_t init_layer;
    TaskQueue* task_queue;
};

__forceinline__ __device__ void loadMatchInfo(
    MatchInfo& shared_info,
    MatchInfo* global_info
)
{
    uint32_t size = sizeof(shared_info);
    if (threadIdx.x < (size + 3) / 4)
    {
        ((uint32_t*)(&shared_info))[threadIdx.x] = ((uint32_t*)(global_info))[threadIdx.x];
    }
    __syncthreads();
}



__forceinline__ __device__ void warpExtend(
    MatchInfo& info,
    uint64_t& warpCounter,
    uintV* warpStack,
    uint32_t* stack_iter,
    uint32_t* stack_size,
    uintV* partial_match,
    uint8_t partial_match_size,
    uint64_t& timeCounter
)
{
    uint8_t current_mapping_vertex = partial_match_size;
    while(true)
    {
        uintV* stack_array = warpStack + current_mapping_vertex * info.stack_layer_size;
        uint8_t parent_start = info.q_row_ptrs[current_mapping_vertex];
        uint8_t parent_num = info.q_row_ptrs[current_mapping_vertex + 1] - parent_start;
        uintV vid_upper_bound = info.q_restriction[current_mapping_vertex] == 0xFF? 0xFFFFFFFFU : partial_match[info.q_restriction[current_mapping_vertex]];
        uintV pivot_vid = partial_match[info.q_cols[parent_start]];
        uintV degree = info.row_ptrs[pivot_vid + 1] - info.row_ptrs[pivot_vid];
        uintV* pivot_start = info.cols + info.row_ptrs[pivot_vid];
        stack_size[current_mapping_vertex] = 0;
        for (uintV i = threadIdinWarp; i < degree; i += 32)
        {
            uintV vid = pivot_start[i];
            if (vid >= vid_upper_bound) break;
            bool not_exist = true;
            for (uint8_t j = 0 ; j < current_mapping_vertex; j ++)
            {
                if (vid == partial_match[j]) {
                    not_exist = false;
                    break;
                }
            }
            __syncwarp(__activemask());
            if (not_exist)
            {
                coalesced_group active = cooperative_groups::coalesced_threads();
                stack_array[stack_size[current_mapping_vertex] + active.thread_rank()] = vid;
            }
            stack_size[current_mapping_vertex] += __reduce_add_sync(__activemask(), not_exist);
        }
        
        for (uint8_t parent = parent_start + 1; parent < parent_start + parent_num; parent++)
        {
            uintV bckNeighbor = partial_match[info.q_cols[parent]];
            BinaryIntersection(stack_array, info.cols + info.row_ptrs[bckNeighbor], stack_size[current_mapping_vertex], info.row_ptrs[bckNeighbor + 1] - info.row_ptrs[bckNeighbor]);
            __syncwarp();
        }
        if (current_mapping_vertex == info.q_vertex_count - 1 || stack_size[current_mapping_vertex] == 0)
        {
            if (threadIdinWarp == 0)
            {
                warpCounter += stack_size[current_mapping_vertex];
            }
            __syncwarp();
            current_mapping_vertex--;
            while (current_mapping_vertex >= partial_match_size && stack_iter[current_mapping_vertex] >= stack_size[current_mapping_vertex] - 1)
            {
                current_mapping_vertex--;
            }
            if (current_mapping_vertex < partial_match_size) break;
            stack_iter[current_mapping_vertex] = stack_iter[current_mapping_vertex] + 1;
            partial_match[current_mapping_vertex] = warpStack[current_mapping_vertex * info.stack_layer_size + stack_iter[current_mapping_vertex]];
            current_mapping_vertex++;
        }
        else
        {
            stack_iter[current_mapping_vertex] = 0;
            partial_match[current_mapping_vertex] = warpStack[(current_mapping_vertex) * info.stack_layer_size];
            current_mapping_vertex++;
        }
    }
    return;
}

__launch_bounds__(1024, 2)
    __global__ void DFSWarpKernel(
        MatchInfo* info,
        uint64_t* global_allocator,
        uint64_t* Counter,
        uint64_t* profileCounter
    )
{
    __shared__ uintV partial_match[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_iter[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_size[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint64_t warpCounter[WARP_PER_BLOCK];
    __shared__ uint64_t timeCounter[WARP_PER_BLOCK];
    __shared__ MatchInfo shared_info;
    warpCounter[warpIdinBlock] = 0;
    timeCounter[warpIdinBlock] = clock();
    loadMatchInfo(shared_info, info);
    uintE init_match_pointer;
    if (threadIdinWarp == 0)
    {
        init_match_pointer = atomicAdd((unsigned long long*)global_allocator, 1UL);
    }
    init_match_pointer = __shfl_sync(0xFFFFFFFF, init_match_pointer, 0);
    while(true)
    {
        if (init_match_pointer >= shared_info.init_task_num) {
            if (threadIdinWarp == 0)
            {
                atomicAdd((unsigned long long*) Counter, warpCounter[warpIdinBlock]);
            }
            break;
        }
        if (threadIdinWarp < shared_info.init_layer)
        {
            partial_match[warpIdinBlock * MAX_QUERY_VERTEX + threadIdinWarp] = shared_info.init_tasks[init_match_pointer * shared_info.init_layer + threadIdinWarp];
        }
        warpExtend(shared_info, 
            warpCounter[warpIdinBlock], 
            shared_info.stacks + warpId * shared_info.q_vertex_count * shared_info.stack_layer_size, 
            stack_iter + warpIdinBlock * MAX_QUERY_VERTEX, 
            stack_size + warpIdinBlock * MAX_QUERY_VERTEX,
            partial_match + warpIdinBlock * MAX_QUERY_VERTEX,
            shared_info.init_layer,
            timeCounter[warpIdinBlock]
        );
        if (threadIdinWarp == 0)
        {
            init_match_pointer = atomicAdd((unsigned long long*)global_allocator, 1UL);
        }
        init_match_pointer = __shfl_sync(0xFFFFFFFF, init_match_pointer, 0);
    }
    timeCounter[warpIdinBlock] = clock() - timeCounter[warpIdinBlock];
    if (threadIdinWarp == 0)
        profileCounter[warpId] = timeCounter[warpIdinBlock];
    // printf("warpId: %d, time: %lu\n", warpId, timeCounter[warpIdinBlock]);
    return;
}


__forceinline__ __device__ void warpExtendOffload(
    MatchInfo& info,
    uint64_t& warpCounter,
    uintV* warpStack,
    uint32_t* stack_iter,
    uint32_t* stack_size,
    uintV* partial_match,
    uint8_t partial_match_size,
    uint64_t& timeCounter
)
{
    uint8_t current_mapping_vertex = partial_match_size;
    timeCounter = clock();
    while(true)
    {
        uint64_t curr_time = clock();
        if (curr_time - timeCounter > 1000000)
        {
            uint8_t offload_layer = partial_match_size;
            while (offload_layer < current_mapping_vertex && stack_iter[offload_layer] == stack_size[offload_layer] - 1)
            {
                offload_layer++;
            }
            if (offload_layer < current_mapping_vertex)
            {
                if (info.task_queue->enqueue(stack_size[offload_layer] - stack_iter[offload_layer] - 1, (uint32_t)(offload_layer + 1), partial_match, warpStack + offload_layer * info.stack_layer_size + stack_iter[offload_layer] + 1))
                {
                    stack_size[offload_layer] = stack_iter[offload_layer] + 1;
                }               
            }
            timeCounter = clock();
        }
        uintV* stack_array = warpStack + current_mapping_vertex * info.stack_layer_size;
        uint8_t parent_start = info.q_row_ptrs[current_mapping_vertex];
        uint8_t parent_num = info.q_row_ptrs[current_mapping_vertex + 1] - parent_start;
        uintV vid_upper_bound = info.q_restriction[current_mapping_vertex] == 0xFF? 0xFFFFFFFFU : partial_match[info.q_restriction[current_mapping_vertex]];
        uintV pivot_vid = partial_match[info.q_cols[parent_start]];
        uintV degree = info.row_ptrs[pivot_vid + 1] - info.row_ptrs[pivot_vid];
        uintV* pivot_start = info.cols + info.row_ptrs[pivot_vid];
        stack_size[current_mapping_vertex] = 0;
        for (uintV i = threadIdinWarp; i < degree; i += 32)
        {
            uintV vid = pivot_start[i];
            if (vid >= vid_upper_bound) break;
            bool not_exist = true;
            for (uint8_t j = 0 ; j < current_mapping_vertex; j ++)
            {
                if (vid == partial_match[j]) {
                    not_exist = false;
                    break;
                }
            }
            __syncwarp(__activemask());
            if (not_exist)
            {
                coalesced_group active = cooperative_groups::coalesced_threads();
                stack_array[stack_size[current_mapping_vertex] + active.thread_rank()] = vid;
            }
            stack_size[current_mapping_vertex] += __reduce_add_sync(__activemask(), not_exist);
        }
        
        for (uint8_t parent = parent_start + 1; parent < parent_start + parent_num; parent++)
        {
            uintV bckNeighbor = partial_match[info.q_cols[parent]];
            BinaryIntersection(stack_array, info.cols + info.row_ptrs[bckNeighbor], stack_size[current_mapping_vertex], info.row_ptrs[bckNeighbor + 1] - info.row_ptrs[bckNeighbor]);
            __syncwarp();
        }
        if (current_mapping_vertex == info.q_vertex_count - 1 || stack_size[current_mapping_vertex] == 0)
        {
            if (threadIdinWarp == 0)
            {
                warpCounter += stack_size[current_mapping_vertex];
            }
            __syncwarp();
            current_mapping_vertex--;
            while (current_mapping_vertex >= partial_match_size && stack_iter[current_mapping_vertex] >= stack_size[current_mapping_vertex] - 1)
            {
                current_mapping_vertex--;
            }
            if (current_mapping_vertex < partial_match_size) break;
            stack_iter[current_mapping_vertex] = stack_iter[current_mapping_vertex] + 1;
            partial_match[current_mapping_vertex] = warpStack[current_mapping_vertex * info.stack_layer_size + stack_iter[current_mapping_vertex]];
            current_mapping_vertex++;
        }
        else
        {
            stack_iter[current_mapping_vertex] = 0;
            partial_match[current_mapping_vertex] = warpStack[(current_mapping_vertex) * info.stack_layer_size];
            current_mapping_vertex++;
        }
    }
    return;
}


__launch_bounds__(1024, 1)
    __global__ void DFSWarpBalanceKernel(
        MatchInfo* info,
        uint64_t* global_allocator,
        uint64_t* Counter,
        uint64_t* profileCounter,
        uint32_t* IdleWarpCounter
    )
{
    __shared__ uintV partial_match[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_iter[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_size[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint64_t warpCounter[WARP_PER_BLOCK];
    __shared__ uint64_t timeCounter[WARP_PER_BLOCK];
    __shared__ uint64_t profiletimeCounter[WARP_PER_BLOCK];
    __shared__ MatchInfo shared_info;
    warpCounter[warpIdinBlock] = 0;
    profiletimeCounter[warpIdinBlock] = clock();
    loadMatchInfo(shared_info, info);
    uintE init_match_pointer;
    if (threadIdinWarp == 0)
    {
        init_match_pointer = atomicAdd((unsigned long long*)global_allocator, 1UL);
    }
    init_match_pointer = __shfl_sync(0xFFFFFFFF, init_match_pointer, 0);
    while(true)
    {
        if (init_match_pointer >= shared_info.init_task_num) {
            // if (threadIdinWarp == 0)
            // {
            //     atomicAdd((unsigned long long*) Counter, warpCounter[warpIdinBlock]);
            // }
            break;
        }
        if (threadIdinWarp < shared_info.init_layer)
        {
            partial_match[warpIdinBlock * MAX_QUERY_VERTEX + threadIdinWarp] = shared_info.init_tasks[init_match_pointer * shared_info.init_layer + threadIdinWarp];
        }
        warpExtendOffload(shared_info, 
            warpCounter[warpIdinBlock], 
            shared_info.stacks + warpId * shared_info.q_vertex_count * shared_info.stack_layer_size, 
            stack_iter + warpIdinBlock * MAX_QUERY_VERTEX, 
            stack_size + warpIdinBlock * MAX_QUERY_VERTEX,
            partial_match + warpIdinBlock * MAX_QUERY_VERTEX,
            shared_info.init_layer,
            timeCounter[warpIdinBlock]
        );
        if (threadIdinWarp == 0)
        {
            init_match_pointer = atomicAdd((unsigned long long*)global_allocator, 1UL);
        }
        init_match_pointer = __shfl_sync(0xFFFFFFFF, init_match_pointer, 0);
    }
    if (threadIdinWarp == 0) {
        atomicAdd(IdleWarpCounter, 1);
    }
    __syncwarp();
    uint64_t cnt = 0;
    while (true)
    {
        cnt ++;
        uint32_t task_size;
        if (shared_info.task_queue->dequeue(task_size, partial_match + warpIdinBlock * MAX_QUERY_VERTEX))
        {
            if (threadIdinWarp == 0)
                atomicSub(IdleWarpCounter, 1);
            __syncwarp();
            warpExtendOffload(shared_info, 
                warpCounter[warpIdinBlock], 
                shared_info.stacks + warpId * shared_info.q_vertex_count * shared_info.stack_layer_size, 
                stack_iter + warpIdinBlock * MAX_QUERY_VERTEX, 
                stack_size + warpIdinBlock * MAX_QUERY_VERTEX,
                partial_match + warpIdinBlock * MAX_QUERY_VERTEX,
                task_size,
                timeCounter[warpIdinBlock]
            );
            if (threadIdinWarp == 0) {
                atomicAdd(IdleWarpCounter, 1);
            }
            __syncwarp();
        }
        else {
            if (*(volatile uint32_t*)IdleWarpCounter == blockDim.x * gridDim.x / 32)
            {
                break;
            }
            else {
                __nanosleep(100);
            }
        }
    }
    if (threadIdinWarp == 0)
    {
        atomicAdd((unsigned long long*) Counter, warpCounter[warpIdinBlock]);
    }
    profiletimeCounter[warpIdinBlock] = clock() - profiletimeCounter[warpIdinBlock];
    if (threadIdinWarp == 0)
        profileCounter[warpId] = profiletimeCounter[warpIdinBlock];
    return;
}

struct ReuseMatchInfo {
    uintE* row_ptrs;
    uintV* cols;
    uintV* stacks;
    uint8_t q_row_ptrs[16];
    uint8_t q_cols[32];
    uint8_t q_reuse_info[16];
    uint8_t q_is_reused[16];
    uint8_t q_restriction[16];
    uintV* init_tasks;
    uintE init_task_num;
    uintV dg_vertex_count;
    uint32_t stack_layer_size;
    uint8_t q_vertex_count;
    uint8_t init_layer;
    TaskQueue* task_queue;
};

__forceinline__ __device__ void loadReuseMatchInfo(
    ReuseMatchInfo& shared_info,
    ReuseMatchInfo* global_info
)
{
    uint32_t size = sizeof(shared_info);
    if (threadIdx.x < (size + 3) / 4)
    {
        ((uint32_t*)(&shared_info))[threadIdx.x] = ((uint32_t*)(global_info))[threadIdx.x];
    }
    __syncthreads();
}

__forceinline__ __device__ void warpExtendReuseOffload(
    ReuseMatchInfo& info,
    uint64_t& warpCounter,
    uintV* warpStack,
    uint32_t* stack_iter,
    uint32_t* stack_size,
    uint32_t* origin_stack_size,
    uintV* partial_match,
    uint8_t partial_match_size,
    uint64_t& timeCounter
)
{
    uint8_t current_mapping_vertex = partial_match_size;
    timeCounter = clock();
    while(true)
    {
        uint64_t curr_time = clock();
        if (curr_time - timeCounter > 1000000)
        {
            uint8_t offload_layer = partial_match_size;
            while (offload_layer < current_mapping_vertex && stack_iter[offload_layer] == stack_size[offload_layer] - 1)
            {
                offload_layer++;
            }
            if (offload_layer < current_mapping_vertex)
            {
                if (info.task_queue->enqueue(stack_size[offload_layer] - stack_iter[offload_layer] - 1, (uint32_t)(offload_layer + 1), partial_match, warpStack + offload_layer * info.stack_layer_size + stack_iter[offload_layer] + 1))
                {
                    stack_size[offload_layer] = stack_iter[offload_layer] + 1;
                }               
            }
            timeCounter = clock();
        }
        uintV* stack_array = warpStack + current_mapping_vertex * info.stack_layer_size;
        uint8_t parent_start = info.q_row_ptrs[current_mapping_vertex];
        uint8_t parent_num = info.q_row_ptrs[current_mapping_vertex + 1] - parent_start;
        uintV vid_upper_bound = info.q_restriction[current_mapping_vertex] == 0xFF? 0xFFFFFFFFU : partial_match[info.q_restriction[current_mapping_vertex]];
        uintV pivot_vid = partial_match[info.q_cols[parent_start] & 0x7f];
        uint8_t reuse_layer = info.q_reuse_info[current_mapping_vertex];
        uintV degree = info.row_ptrs[pivot_vid + 1] - info.row_ptrs[pivot_vid];
        uintV* pivot_start = info.cols + info.row_ptrs[pivot_vid];
        bool need_reuse = reuse_layer != 0xFF && reuse_layer >= partial_match_size;
        bool is_reused = info.q_is_reused[current_mapping_vertex];
        if (need_reuse) {
            pivot_start = warpStack + reuse_layer * info.stack_layer_size;
            degree = origin_stack_size[reuse_layer];
        }
        origin_stack_size[current_mapping_vertex] = 0;
        stack_size[current_mapping_vertex] = 0;
        for (uintV i = threadIdinWarp; i < degree; i += 32)
        {
            uintV vid = pivot_start[i];
            if (!is_reused && vid >= vid_upper_bound) break;
            bool not_exist = true;
            for (uint8_t j = 0 ; j < current_mapping_vertex; j ++)
            {
                if (vid == partial_match[j]) {
                    not_exist = false;
                    break;
                }
            }
            __syncwarp(__activemask());
            if (not_exist)
            {
                coalesced_group active = cooperative_groups::coalesced_threads();
                stack_array[origin_stack_size[current_mapping_vertex] + active.thread_rank()] = vid;
            }
            origin_stack_size[current_mapping_vertex] += __reduce_add_sync(__activemask(), not_exist);
            stack_size[current_mapping_vertex] += __reduce_add_sync(__activemask(), not_exist && vid < vid_upper_bound);
        }
        __syncwarp();
        for (uint8_t parent = parent_start + 1; parent < parent_start + parent_num; parent++)
        {
            if (need_reuse && (info.q_cols[parent] & 0x80)) continue;
            uintV bckNeighbor = partial_match[info.q_cols[parent] & 0x7F];
            BinaryIntersection(stack_array, info.cols + info.row_ptrs[bckNeighbor], origin_stack_size[current_mapping_vertex], stack_size[current_mapping_vertex],
                 info.row_ptrs[bckNeighbor + 1] - info.row_ptrs[bckNeighbor], vid_upper_bound);
            __syncwarp();
        }

        if (current_mapping_vertex == info.q_vertex_count - 1 || stack_size[current_mapping_vertex] == 0)
        {
            if (threadIdinWarp == 0)
            {
                warpCounter += stack_size[current_mapping_vertex];
            }
            __syncwarp();
            current_mapping_vertex--;
            while (current_mapping_vertex >= partial_match_size && (stack_iter[current_mapping_vertex] >= stack_size[current_mapping_vertex] - 1))
            {
                current_mapping_vertex--;
            }
            if (current_mapping_vertex < partial_match_size) break;
            stack_iter[current_mapping_vertex] = stack_iter[current_mapping_vertex] + 1;
            partial_match[current_mapping_vertex] = warpStack[current_mapping_vertex * info.stack_layer_size + stack_iter[current_mapping_vertex]];
            current_mapping_vertex++;
        }
        else
        {
            stack_iter[current_mapping_vertex] = 0;
            partial_match[current_mapping_vertex] = warpStack[(current_mapping_vertex) * info.stack_layer_size];
            current_mapping_vertex++;
        }
    }
    return;
}


__launch_bounds__(1024, 1)
    __global__ void DFSWarpReuseBalanceKernel(
        ReuseMatchInfo* info,
        uint64_t* global_allocator,
        uint64_t* Counter,
        uint64_t* profileCounter,
        uint32_t* IdleWarpCounter
    )
{
    __shared__ uintV partial_match[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_iter[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t stack_size[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint32_t origin_stack_size[WARP_PER_BLOCK * MAX_QUERY_VERTEX];
    __shared__ uint64_t warpCounter[WARP_PER_BLOCK];
    __shared__ uint64_t timeCounter[WARP_PER_BLOCK];
    __shared__ uint64_t profiletimeCounter[WARP_PER_BLOCK];
    __shared__ ReuseMatchInfo shared_info;
    warpCounter[warpIdinBlock] = 0;
    profiletimeCounter[warpIdinBlock] = clock();
    loadReuseMatchInfo(shared_info, info);
    uintE init_match_pointer;
    if (threadIdinWarp == 0)
    {
        init_match_pointer = atomicAdd((unsigned long long*)global_allocator, 1UL);
    }
    init_match_pointer = __shfl_sync(0xFFFFFFFF, init_match_pointer, 0);
    while(true)
    {
        if (init_match_pointer >= shared_info.init_task_num) {
            // if (threadIdinWarp == 0)
            // {
            //     atomicAdd((unsigned long long*) Counter, warpCounter[warpIdinBlock]);
            // }
            break;
        }
        if (threadIdinWarp < shared_info.init_layer)
        {
            partial_match[warpIdinBlock * MAX_QUERY_VERTEX + threadIdinWarp] = shared_info.init_tasks[init_match_pointer * shared_info.init_layer + threadIdinWarp];
        }
        warpExtendReuseOffload(shared_info, 
            warpCounter[warpIdinBlock], 
            shared_info.stacks + warpId * shared_info.q_vertex_count * shared_info.stack_layer_size, 
            stack_iter + warpIdinBlock * MAX_QUERY_VERTEX, 
            stack_size + warpIdinBlock * MAX_QUERY_VERTEX,
            origin_stack_size + warpIdinBlock * MAX_QUERY_VERTEX,
            partial_match + warpIdinBlock * MAX_QUERY_VERTEX,
            shared_info.init_layer,
            timeCounter[warpIdinBlock]
        );
        if (threadIdinWarp == 0)
        {
            init_match_pointer = atomicAdd((unsigned long long*)global_allocator, 1UL);
        }
        init_match_pointer = __shfl_sync(0xFFFFFFFF, init_match_pointer, 0);
    }
    if (threadIdinWarp == 0) {
        atomicAdd(IdleWarpCounter, 1);
    }
    __syncwarp();
    uint64_t cnt = 0;
    while (true)
    {
        cnt ++;
        uint32_t task_size;
        if (shared_info.task_queue->dequeue(task_size, partial_match + warpIdinBlock * MAX_QUERY_VERTEX))
        {
            if (threadIdinWarp == 0)
                atomicSub(IdleWarpCounter, 1);
            __syncwarp();
            warpExtendReuseOffload(shared_info, 
                warpCounter[warpIdinBlock], 
                shared_info.stacks + warpId * shared_info.q_vertex_count * shared_info.stack_layer_size, 
                stack_iter + warpIdinBlock * MAX_QUERY_VERTEX, 
                stack_size + warpIdinBlock * MAX_QUERY_VERTEX, 
                origin_stack_size + warpIdinBlock * MAX_QUERY_VERTEX,
                partial_match + warpIdinBlock * MAX_QUERY_VERTEX,
                task_size,
                timeCounter[warpIdinBlock]
            );
            if (threadIdinWarp == 0) {
                atomicAdd(IdleWarpCounter, 1);
            }
            __syncwarp();
        }
        else {
            if (*(volatile uint32_t*)IdleWarpCounter == blockDim.x * gridDim.x / 32)
            {
                break;
            }
            else {
                __nanosleep(100);
            }
        }
    }
    if (threadIdinWarp == 0)
    {
        atomicAdd((unsigned long long*) Counter, warpCounter[warpIdinBlock]);
    }
    profiletimeCounter[warpIdinBlock] = clock() - profiletimeCounter[warpIdinBlock];
    if (threadIdinWarp == 0)
        profileCounter[warpId] = profiletimeCounter[warpIdinBlock];
    return;
}