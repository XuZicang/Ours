#pragma once
#include "Meta.h"
#include "utils/GPUUtil.cuh"
#include "utils/DeviceArray.cuh"
#include <cstdint>
// #define EMPTYSLOT 0xFFFFFFFFU
alignas(64) struct Task
{
    uint32_t vertex_num;
    uintV partial_match[15];
};

struct WorkerQueue
{
    volatile bool *writeFlag;
    volatile Task *informations;
    // volatile uint64_t head;
    volatile uint64_t tail;
    volatile uint64_t write_pointer;
    volatile uint32_t queueCount;
    int mutex;
    uint32_t capacity; // set as maximum warp
};

WorkerQueue *init_device_WorkerQueue(uint32_t capacity, CudaContext *context)
{
    WorkerQueue cpu_queue;
    // cpu_queue.head = 0;
    cpu_queue.tail = 0;
    cpu_queue.write_pointer = 0;
    cpu_queue.queueCount = 0;
    cpu_queue.mutex = 0;
    cpu_queue.capacity = capacity;
    DeviceArray<bool> *write_flag = new DeviceArray<bool>(capacity, context);
    DeviceArray<Task> *informations = new DeviceArray<Task>(capacity, context);
    CUDA_ERROR(cudaMemset(write_flag->GetArray(), 0, capacity * sizeof(bool)));
    cpu_queue.writeFlag = write_flag->GetArray();
    cpu_queue.informations = informations->GetArray();
    // WorkerQueue *gpu_queue;
    DeviceArray<WorkerQueue> *dev_queue = new DeviceArray<WorkerQueue>(1, context);
    HToD(dev_queue->GetArray(), &cpu_queue, 1);
    return dev_queue->GetArray();
}

__forceinline__ __device__ void enqueue(WorkerQueue &queue, uint32_t &pos)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        pos = atomicAdd((unsigned long long *)&queue.tail, 1UL) % queue.capacity;
        atomicAdd((unsigned *)&queue.queueCount, 1U);
    }
    __threadfence();
    __syncthreads();
}

__forceinline__ __device__ void dequeue(WorkerQueue &queue, uint32_t pos)
{
    if (threadIdx.x == 0)
    {
        queue.writeFlag[pos] = false;
        // atomicAdd((unsigned long long*)&queue.head, 1UL);
    }
    __threadfence();
    __syncthreads();
}
__forceinline__ __device__ bool inform(WorkerQueue &queue, uintV *partial_match, uint8_t current_layer)
{
    uint64_t pos;
    if (threadIdinWarp == 0)
    {
        if (atomicCAS(&queue.mutex, 0, 1) == 1)
        {
            pos = 0xFFFFFFFFFFFFFFFFUL;
        }
        else
        {
            pos = queue.write_pointer;
            if (pos < queue.tail)
            {
                queue.write_pointer += 1;
            }
            else
            {
                pos = 0xFFFFFFFFFFFFFFFFUL;
            }
            __threadfence();
            atomicExch(&queue.mutex, 0);
        }
    }
    pos = __shfl_sync(0xFFFFFFFF, pos, 0);
    if (pos == 0xFFFFFFFFFFFFFFFFUL)
        return false;
    else
    {
        pos = pos % queue.capacity;
    }
    if (threadIdinWarp < current_layer)
    {
        queue.informations[pos].vertex_num = current_layer;
        queue.informations[pos].partial_match[threadIdinWarp] = partial_match[threadIdinWarp];
    }
    __threadfence();
    __syncwarp();
    if (threadIdinWarp == 0)
    {
        atomicSub((unsigned *)&queue.queueCount, 1U);
        queue.writeFlag[pos] = true;
    }
    __threadfence();
    __syncwarp();
    return true;
}
// __forceinline__ __device__ bool inform(WorkerQueue& queue, uintV* partial_match, uint8_t current_layer)
// {
//     uint64_t pos;
//     if (threadIdinWarp == 0)
//     {
//         while(atomicCAS(&queue.mutex, 0, 1) == 1)
//         {
//             __nanosleep(20);
//         }
//         pos = queue.write_pointer;
//         if (pos < queue.tail)
//         {
//             queue.write_pointer += 1;
//         }
//         else
//         {
//             pos = 0xFFFFFFFFFFFFFFFFUL;
//         }
//         __threadfence();
//         atomicExch(&queue.mutex, 0);
//     }
//     pos = __shfl_sync(0xFFFFFFFF, pos, 0);
//     if (pos == 0xFFFFFFFFFFFFFFFFUL) return false;
//     else {
//         pos = pos % queue.capacity;
//     }
//     if (threadIdinWarp < current_layer)
//     {
//         queue.informations[pos].vertex_num = current_layer;
//         queue.informations[pos].partial_match[threadIdinWarp] = partial_match[threadIdinWarp];
//     }
//     __threadfence();
//     __syncwarp();
//     if (threadIdinWarp == 0) {
//         atomicSub((unsigned *)&queue.queueCount, 1U);
//         queue.writeFlag[pos] = true;
//     }
//     __threadfence();
//     __syncwarp();
//     return true;
// }

__forceinline__ __device__ bool waitTask(WorkerQueue &queue, uint32_t pos, uintV *partial_match, uint32_t &vertex_num)
{
    while (true)
    {
        if (queue.queueCount == gridDim.x)
            return false;
        if (queue.writeFlag[pos] == true)
        {
            vertex_num = queue.informations[pos].vertex_num;
            if (threadIdinWarp < vertex_num)
                partial_match[threadIdinWarp] = queue.informations[pos].partial_match[threadIdinWarp];
            break;
        }
        __nanosleep(100);
    }
    __syncthreads();
    return true;
}
