#ifndef _TASK_QUEUE_CUH
#define _TASK_QUEUE_CUH
#include "Meta.h"
#include <cuda.h>
#include "utils/GPUUtil.cuh"
const uint32_t UNRELEASED = 0xFFFFFFFFU;
const uint32_t UNFILLED = 0x0U;

__align__(32) struct Task1 {
    uint32_t size;
    uintV partial_match[7];
};
class TaskQueue {
public:
    __forceinline__ __device__ bool enqueue(uint32_t num, uint32_t size, uintV* partial_match, uintV* tails)
    {
        assert((int)size >= 2);
        int fill;
        if (threadIdinWarp == 0)
            fill = atomicAdd(&task_count, (int)num);
        fill = __shfl_sync(0xFFFFFFFF, fill, 0);
        if (fill + num <= queue_size)
        {
            uint32_t pos;
            if (threadIdinWarp == 0)
                pos = atomicAdd(&tail, num) % queue_size;
            pos = __shfl_sync(0xFFFFFFFF, pos, 0);
            for (uint32_t i = 0; i < num; i ++)
            {
                if (threadIdinWarp == 0) {
                    while(atomicCAS(&tasks[(pos + i) % queue_size].size, UNFILLED, UNRELEASED) != UNFILLED) {
                        __nanosleep(10);
                    }
                }
                __syncwarp(__activemask());
                if (threadIdinWarp < size - 1)
                    tasks[(pos + i) % queue_size].partial_match[threadIdinWarp] = partial_match[threadIdinWarp];
            }
            for (uint32_t i = threadIdinWarp; i < num; i += 32)
            {
                tasks[(pos + i) % queue_size].partial_match[size - 1] = tails[i];
                tasks[(pos + i) % queue_size].size = size;
            }
            __threadfence();
            __syncwarp(__activemask());
            return true;
        }
        else
        {
            if (threadIdinWarp == 0)
                atomicSub(&task_count, (int) num);
            __syncwarp(__activemask());
            return false;
        }
    }
    __forceinline__ __device__ bool dequeue(uint32_t& size, uintV* partial_match)
    {
        int readable;
        if (threadIdinWarp == 0) {
            readable = atomicSub(&task_count, 1);
        }
        readable = __shfl_sync(0xFFFFFFFFU, readable, 0);
        if (readable <= 0)
        {
            if (threadIdinWarp == 0)
                atomicAdd(&task_count, 1);
            return false;
        }
        uint32_t pos;
        if (threadIdinWarp == 0)
            pos = atomicAdd(&head, 1) % queue_size;
        pos = __shfl_sync(0xFFFFFFFF, pos, 0);
        if (threadIdinWarp == 0) {
            while (true) {
                size = *(volatile uint32_t*) &tasks[pos].size;
                if (size != UNFILLED && size != UNRELEASED) break;
                __nanosleep(10);
            }
        }
        size = __shfl_sync(0xFFFFFFFF, size, 0);
        if (threadIdinWarp < size)
        {
            partial_match[threadIdinWarp] = tasks[pos].partial_match[threadIdinWarp];
        }
        if (threadIdinWarp == 0)
            tasks[pos].size = UNFILLED;
        __threadfence();
        __syncwarp(__activemask());
        return true;
    }
    Task1* tasks;
    uint32_t head;
    uint32_t tail;
    int task_count;
    int queue_size;
};
#endif