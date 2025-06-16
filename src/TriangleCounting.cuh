#pragma once
#include "Meta.h"
#include <cuda.h>
#include "utils/GPUUtil.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "Balance.cuh"
#include "Matchutil.cuh"
#include "Cuckoo.cuh"
using namespace cooperative_groups;
// using cooperative_groups::thread_group;

#define CHUNK_SIZE 8
__forceinline__ __device__ uintV BinaryIntersectionCount(__restrict__ const uintV *arrayA, __restrict__ const uintV *arrayB, uintV sizeA, uintV sizeB)
{
    uintV write_pos = 0;
    for (uintV i = threadIdinWarp; i < sizeA; i += 32)
    {
        bool is_exist = BinarySearch(arrayA[i], arrayB, sizeB);
        write_pos += __reduce_add_sync(__activemask(), is_exist);
    }
    return write_pos;
}

__forceinline__ __device__ uintV BinaryIntersectionCountAligned(__restrict__ const uintV *arrayA, __restrict__ const uintV *arrayB, uintV sizeA, uintV sizeB)
{
    uintV write_pos = 0;
#pragma unroll 4
    for (uintV i = threadIdinWarp; i < sizeA; i += 32)
    {
        bool is_exist = (arrayA[i] != 0xFFFFFFFF) && BinarySearch(arrayA[i], arrayB, sizeB);
        write_pos += __reduce_add_sync(__activemask(), is_exist);
    }
    return write_pos;
}

__launch_bounds__(1024)
    __global__ void triangle_counting(
        uintV vertex_count,
        uintE *row_ptrs,
        uintV *cols,
        uint64_t *Counters,
        uintV *global_allocator,
        uintV chunk_size)
{
    __shared__ uint64_t warpCounter[32];
    warpCounter[warpIdinBlock] = 0;
    while (true)
    {
        uintV chunk_start;
        if (threadIdinWarp == 0)
        {
            chunk_start = atomicAdd(global_allocator, chunk_size);
        }
        chunk_start = __shfl_sync(0xFFFFFFFF, chunk_start, 0);
        __syncwarp();
        if (chunk_start > vertex_count)
        {
            if (threadIdinWarp == 0)
                atomicAdd((unsigned long long *)Counters, (unsigned long long)warpCounter[warpIdinBlock]);
            break;
        }
        uintV chunk_end = min(vertex_count, chunk_start + chunk_size);
        for (uintV u = chunk_start; u < chunk_end; u++)
        {
            for (uintE j = row_ptrs[u]; j < row_ptrs[u + 1]; j++)
            {
                // intersect N(u), N(cols[j])
                uintV v = cols[j];
                uintV count = BinaryIntersectionCount(cols + row_ptrs[u], cols + row_ptrs[v], j - row_ptrs[u], row_ptrs[v + 1] - row_ptrs[v]);
                if (threadIdinWarp == 0)
                    warpCounter[warpIdinBlock] += count;
            }
        }
    }
}

__forceinline__ __device__ void load_neighbor_list(uintV *dst, const uintV *src, uintV src_size)
{
    for (uint32_t i = threadIdx.x; i < src_size; i += blockDim.x)
    {
        dst[i] = src[i];
    }
    __syncthreads();
}

__forceinline__ __device__ void load_neighbor_list_warp(uintV *dst, uintV *src, uintV src_size)
{
    dst[threadIdinWarp] = src[threadIdinWarp];
}

__forceinline__ __device__ void load_neighbor_list_group(uintV *dst, uintV *src, uintV src_size, thread_group group)
{
    for (uint32_t i = group.thread_rank(); i < src_size; i += group.size())
    {
        dst[i] = src[i];
    }
    group.sync();
}

__launch_bounds__(512, 4)
    __global__ void triangle_counting_with_shared_memory(
        uintV vertex_count,
        uintE *row_ptrs,
        uintV *cols,
        uintV *dense_root,
        uintV dense_root_num,
        uint64_t *Counters,
        uintV *global_allocator,
        uintV chunk_size)
{
    __shared__ uintV shared_list[2 * 1024];
    __shared__ uint64_t warpCounter[16];
    __shared__ uintV chunk_start;
    __shared__ uintV chunk_end;
    // __shared__ uint64_t clock;
    // if (threadIdx.x == 0)
    //     clock = clock64();
    // __syncthreads();
    warpCounter[warpIdinBlock] = 0;
    while (true)
    {
        if (threadIdx.x == 0)
        {
            chunk_start = atomicAdd(global_allocator, chunk_size);
            chunk_end = min(dense_root_num, chunk_start + chunk_size);
        }
        __syncthreads();
        if (chunk_start > dense_root_num)
        {
            if (threadIdinWarp == 0)
                atomicAdd((unsigned long long *)Counters, (unsigned long long)warpCounter[warpIdinBlock]);
            break;
        }
        for (uintV i = chunk_start; i < chunk_end; i++)
        {
            uintV u = dense_root[i];
            uintV degree = row_ptrs[u + 1] - row_ptrs[u];
            if (degree <= 2048)
            load_neighbor_list(shared_list, cols + row_ptrs[u], degree);

            for (uint32_t j = warpIdinBlock + 1; j < degree; j += 16)
            {
                uintV v = degree <= 2048? shared_list[j] : cols[row_ptrs[u] + j];
                if (v == 0xFFFFFFFFU)
                    break;
                // printf("%d %d\n", u, v);
                uintV count = BinaryIntersectionCountAligned(shared_list, cols + row_ptrs[v], j, row_ptrs[v + 1] - row_ptrs[v]);
                // uintV count = BinaryIntersectionCountAligned(cols + row_ptrs[v], shared_list, row_ptrs[v + 1] - row_ptrs[v], row_ptrs[u + 1] - row_ptrs[u]);
                if (threadIdinWarp == 0)
                {
                    warpCounter[warpIdinBlock] += count;
                }
            }
            __syncthreads();
        }
    }
    // if (threadIdx.x == 0) {
    //     clock = clock64() - clock;
    //     printf("%d %lu\n", blockIdx.x, clock);
    // }
    // __syncthreads();
    return;
}

__global__ void
    __launch_bounds__(1024, 2)
        triangle_counting_sparse_with_subwarp(
            uintV vertex_count,
            uintE *row_ptrs,
            uintV *cols,
            uintV *sparse_root, // root with neighbors less than 32
            uintV sparse_root_num,
            uint64_t *Counters,
            uintV *global_allocator,
            uintV chunk_size)
{
    __shared__ uint64_t warpCounter[32];
    __shared__ uintV shared_list[32 * 32];
    __shared__ uintV chunk_start[32];
    __shared__ uintV chunk_end[32];
    warpCounter[warpIdinBlock] = 0;
    while (true)
    {
        if (threadIdinWarp == 0)
        {
            chunk_start[warpIdinBlock] = atomicAdd(global_allocator, chunk_size);
            chunk_end[warpIdinBlock] = min(sparse_root_num, chunk_start[warpIdinBlock] + chunk_size);
        }
        __syncwarp();
        if (chunk_start[warpIdinBlock] > sparse_root_num)
        {
            if (threadIdinWarp == 0)
                atomicAdd((unsigned long long *)Counters, (unsigned long long)warpCounter[warpIdinBlock]);
            break;
        }
        for (uintV i = chunk_start[warpIdinBlock]; i < chunk_end[warpIdinBlock]; i++)
        {
            uintV u = sparse_root[i];
            uintV neighbor_num = row_ptrs[u + 1] - row_ptrs[u];
            if (threadIdinWarp < neighbor_num)
                shared_list[warpIdinBlock * 32 + threadIdinWarp] = cols[row_ptrs[u] + threadIdinWarp];
            for (uintE j = 1; j < neighbor_num; j++)
            {
                uintV v = shared_list[warpIdinBlock * 32 + j];
                if (v == 0xFFFFFFFFU)
                    break;
                uintV count = BinaryIntersectionCountAligned(shared_list + warpIdinBlock * 32, cols + row_ptrs[v], j, row_ptrs[v + 1] - row_ptrs[v]);
                // uintV count = BinaryIntersectionCountAligned(cols + row_ptrs[v], shared_list + warpIdinBlock * 32, row_ptrs[v + 1] - row_ptrs[v],  neighbor_num);
                if (threadIdinWarp == 0)
                    warpCounter[warpIdinBlock] += count;
                __syncwarp();
            }
        }
    }
}

__global__ void
    __launch_bounds__(512, 4)
        triangle_counting_with_shared_memory_unroll(
            uintV vertex_count,
            uintE *row_ptrs,
            uintV *cols,
            uintV *dense_root,
            uintV dense_root_num,
            uint64_t *Counters,
            uintV *global_allocator,
            uintV chunk_size)
{
    __shared__ uintV shared_list[2 * 1024];
    __shared__ uint64_t warpCounter[16];
    __shared__ uintV chunk_start;
    __shared__ uintV chunk_end;
    warpCounter[warpIdinBlock] = 0;
    while (true)
    {
        if (threadIdx.x == 0)
        {
            chunk_start = atomicAdd(global_allocator, chunk_size);
            chunk_end = min(dense_root_num, chunk_start + chunk_size);
        }
        __syncthreads();
        if (chunk_start > dense_root_num)
        {
            if (threadIdinWarp == 0)
                atomicAdd((unsigned long long *)Counters, (unsigned long long)warpCounter[warpIdinBlock]);
            break;
        }
        for (uintV i = chunk_start; i < chunk_end; i++)
        {
            uintV u = dense_root[i];
            uintV degree = row_ptrs[u + 1] - row_ptrs[u];
            load_neighbor_list(shared_list, cols + row_ptrs[u], degree);
            for (uint32_t j = threadIdx.x; j < degree * (degree - 1) / 2; j += blockDim.x)
            {
                uint32_t r = uint32_t((1 + sqrtf(float(1 + 8 * j))) / 2);
                uintV c = shared_list[j - r * (r - 1) / 2];
                // if (!(int(j - r * (r - 1) / 2) >= 0 && int(j - r * (r - 1) / 2) < degree && int(j - r * (r - 1) / 2) < r))
                // {
                //     printf("%d %d\n", j, r, j - r * (r - 1) / 2);
                // }
                uintV v = shared_list[r];
                // if (c == 0xFFFFFFFFU || v == 0xFFFFFFFFU)
                // {
                //     continue;
                // }
                bool exists = c != 0xFFFFFFFFU && v != 0xFFFFFFFFU && BinarySearch(c, cols + row_ptrs[v], row_ptrs[v + 1] - row_ptrs[v]);
                // __syncwarp(__activemask());
                uintV count = __reduce_add_sync(__activemask(), exists);
                if (threadIdinWarp == 0)
                {
                    warpCounter[threadIdinWarp] += count;
                }
            }
            __syncthreads();
        }
    }
    return;
}

__global__ void
    __launch_bounds__(512, 4)
        triangle_counting_with_shared_memory_cuckoo(
            uintV vertex_count,
            uintE *row_ptrs,
            uintV *cols,
            uintV *dense_root,
            uintV dense_root_num,
            uint32_t *hash_table_sizes,
            uintV *hash_tables,
            uint64_t *Counters,
            uintV *global_allocator,
            uintV chunk_size)
{
    // __shared__ __align__(128) uintV shared_list[2 * 1024];
    __shared__ __align__(128) uintV shared_hash_table0[2 * 1024];
    __shared__ __align__(128) uintV shared_hash_table1[2 * 1024];
    // __shared__ uint64_t warpCounter[16];
    __shared__ uintV chunk_start;
    __shared__ uintV chunk_end;
    uint32_t local_count = 0;
    // __shared__ uint64_t clock;
    // if (threadIdx.x == 0)
    //     clock = clock64();
    // __syncthreads();
    while (true)
    {
        if (threadIdx.x == 0)
        {
            chunk_start = atomicAdd(global_allocator, chunk_size);
            chunk_end = min(dense_root_num, chunk_start + chunk_size);
        }
        __syncthreads();
        if (chunk_start > dense_root_num)
        {
            local_count = __reduce_add_sync(0xFFFFFFFFU, local_count);
            // if (threadIdinWarp == 0) atomicAdd((unsigned long long *)Counters, (unsigned long long)warpCounter[warpIdinBlock]);
            if (threadIdinWarp == 0)
                atomicAdd((unsigned long long *)Counters, (unsigned long long)local_count);
            break;
        }
        for (uintV i = chunk_start; i < chunk_end; i++)
        {
            const uintV u = dense_root[i];
            const uintE u_start = row_ptrs[u];
            const uintE u_end = row_ptrs[u + 1];
            const uintV u_degree = u_end - u_start;
            // load_neighbor_list(shared_list, cols + u_start, u_degree);
            // construct_hash_table(shared_hash_table, cols + row_ptrs[u], degree);

            const uint32_t table_start = hash_table_sizes[i];
            const uint32_t table_total_size = hash_table_sizes[i + 1] - table_start;
            uint32_t current_table_size = table_total_size / 2;
            const uintV *global_table0 = hash_tables + table_start;
            const uintV *global_table1 = hash_tables + table_start + current_table_size;
            if (current_table_size <= 2048)
            {
                // 使用共享内存
                for (uintV idx = threadIdx.x; idx < current_table_size; idx += blockDim.x)
                {
                    shared_hash_table0[idx] = global_table0[idx];
                    shared_hash_table1[idx] = global_table1[idx];
                }
            }
            __syncthreads();
            for (uint32_t j = warpIdinBlock + 1; j < u_degree; j += 16)
            {
                const uintV v = cols[u_start + j];
                if (v == 0xFFFFFFFFU)
                    continue;

                // 获取v的邻居信息
                const uintE v_start = row_ptrs[v];
                const uintE v_end = row_ptrs[v + 1];
                const uintV v_degree = v_end - v_start;
                for (uintV k = threadIdinWarp; k < v_degree; k += 32)
                {
                    const uintV w = cols[v_start + k];
                    // if (w <= v) continue;  // 避免重复计数

                    // 哈希表查找（优先使用共享内存）
                    uint32_t h0 = hash0(w) & (current_table_size - 1);
                    uint32_t h1 = hash1(w) & (current_table_size - 1);

                    bool found = false;
                    if (current_table_size <= 2048)
                    {
                        found = (shared_hash_table0[h0] == w) ||
                                (shared_hash_table1[h1] == w);
                    }
                    else
                    {
                        found = (global_table0[h0] == w) ||
                                (global_table1[h1] == w);
                    }
                    local_count += found ? 1 : 0;
                }
            }
            __syncthreads();
        }
    }
    return;
}

__forceinline__ __device__ bool HashSearch(uintV vid, const uintV *hash_table, uint32_t table_size)
{
    uint32_t bin_id = hash1(vid) % 32;
    for (int j = 0; j < table_size; j++)
    {
        uintV table_v = hash_table[j * 32 + bin_id];
        if (table_v == vid)
            return true;
        if (table_v > vid)
            return false;
        if (table_v == 0xFFFFFFFFU)
            return false;
    }
    return false;
}

__forceinline__ __device__ uint32_t HashIntersectionCount(uintV *array, uintV array_size, const uintV *hash_table, uint32_t table_size)
{
    uint32_t count = 0;
    for (int i = threadIdinWarp; i < array_size; i += 32)
    {
        uintV vid = array[i];
        bool exist = false;
        if (vid != 0xFFFFFFFF)
            exist = HashSearch(vid, hash_table, table_size);
        count += __reduce_add_sync(__activemask(), exist);
    }
    return count;
}

__global__ void
    __launch_bounds__(512, 4)
        triangle_counting_with_hash(
            uintV vertex_count,
            uintE *row_ptrs,
            uintV *cols,
            uintV *dense_root,
            uintV dense_root_num,
            uint32_t *hash_table_sizes,
            uintV *hash_tables,
            uint64_t *Counters,
            uintV *global_allocator,
            uintV chunk_size)
{
    __shared__ uintV shared_list[2 * 1024];
    // __shared__ uintV hash_table[32 * 32 * 4];
    // __shared__ uint32_t bin_pointer[32 * 16];
    __shared__ uint64_t warpCounter[16];
    __shared__ uintV chunk_start;
    __shared__ uintV chunk_end;
    warpCounter[warpIdinBlock] = 0;
    // __shared__ uint64_t clock;
    // if (threadIdx.x == 0)
    //     clock = clock64();
    // __syncthreads();
    while (true)
    {
        if (threadIdx.x == 0)
        {
            chunk_start = atomicAdd(global_allocator, chunk_size);
            chunk_end = min(dense_root_num, chunk_start + chunk_size);
        }
        __syncthreads();
        if (chunk_start > dense_root_num)
        {
            if (threadIdinWarp == 0)
                atomicAdd((unsigned long long *)Counters, (unsigned long long)warpCounter[warpIdinBlock]);
            break;
        }
        for (uintV i = chunk_start; i < chunk_end; i++)
        {
            uintV u = dense_root[i];
            uintV degree = row_ptrs[u + 1] - row_ptrs[u];
            load_neighbor_list(shared_list, cols + row_ptrs[u], degree);

            for (uint32_t j = warpIdinBlock + 1; j < degree; j += 16)
            {
                uintV v = shared_list[j];
                if (v == 0xFFFFFFFFU)
                    break;
                // printf("%d %d\n", u, v);
                uintV count = HashIntersectionCount(shared_list, j, hash_tables + 32 * hash_table_sizes[v], hash_table_sizes[v + 1] - hash_table_sizes[v]);
                // if (warpId == 0)
                // printf("%u %u\n", hash_table_sizes[v + 1] - hash_table_sizes[v], count);
                // uintV count = BinaryIntersectionCountAligned(cols + row_ptrs[v], shared_list, row_ptrs[v + 1] - row_ptrs[v], row_ptrs[u + 1] - row_ptrs[u]);
                if (threadIdinWarp == 0)
                {
                    warpCounter[warpIdinBlock] += count;
                }
            }
            __syncthreads();
        }
    }
}

__forceinline__ __device__ uint32_t HashIntersectionWithPointer(const uintV *hash_table, uint32_t table_size, const uintV *array, uint32_t array_size, uint32_t *bin_pointers)
{
    uint32_t count = 0;
    bin_pointers[threadIdinWarp] = 0;
    for (int i = threadIdinWarp; i < array_size; i += 32)
    {
        uintV vid = array[i];
        if (vid == 0xFFFFFFFFU) break;
        uint32_t bin_id = hash1(vid) & 0x1F;
        for (; bin_pointers[bin_id] < table_size; bin_pointers[bin_id] = bin_pointers[bin_id] + 1)
        {
            if (hash_table[(bin_pointers[bin_id] << 5) + bin_id] >= vid)
            {
                count += hash_table[(bin_pointers[bin_id] << 5) + bin_id] == vid;
                break;
            }
        }
        __syncwarp(__activemask());
    }
    count = __reduce_add_sync(0xFFFFFFFFU, count);
    return count;
}
__forceinline__ __device__ uint32_t HashIntersectionWithPointer1(const uintV *hash_table, uint32_t table_size, const uintV *array, uint32_t array_size, uint32_t *bin_pointers)
{
    uint32_t count = 0;
    bin_pointers[threadIdinWarp] = 0;
#pragma unroll 4
    for (int i = threadIdinWarp; i < array_size; i += 32)
    {
        uintV vid = array[i];
        if (vid == 0xFFFFFFFFU) break;
        uint32_t bin_id = hash1(vid) & 0x1F;
        for (; bin_pointers[bin_id] < table_size; bin_pointers[bin_id] = bin_pointers[bin_id] + 1)
        {
            if (hash_table[(bin_pointers[bin_id] << 5) + bin_id] >= vid)
            {
                count += hash_table[(bin_pointers[bin_id] << 5) + bin_id] == vid;
                break;
            }
        }
        __syncwarp(__activemask());
    }
    // count = __reduce_add_sync(0xFFFFFFFFU, count);
    return count;
}
__global__ void
    __launch_bounds__(512, 4)
        triangle_counting_with_hash1(
            uintV vertex_count,
            uintE *row_ptrs,
            uintV *cols,
            uintV *dense_root,
            uintV dense_root_num,
            uint32_t *hash_table_sizes,
            uintV *hash_tables,
            uint64_t *Counters,
            uintV *global_allocator,
            uintV chunk_size)
{
    __shared__ uintV shared_list[2 * 1024];
    __shared__ uintV hash_table[32 * 128];
    __shared__ uint32_t bin_pointer[32 * 16];
    __shared__ uint64_t warpCounter[16];
    __shared__ uintV chunk_start;
    __shared__ uintV chunk_end;
    warpCounter[warpIdinBlock] = 0;
    while (true)
    {
        if (threadIdx.x == 0)
        {
            chunk_start = atomicAdd(global_allocator, chunk_size);
            chunk_end = min(dense_root_num, chunk_start + chunk_size);
        }
        __syncthreads();
        if (chunk_start >= dense_root_num)
        {
            if (threadIdinWarp == 0)
                atomicAdd((unsigned long long *)Counters, (unsigned long long)warpCounter[warpIdinBlock]);
            // printf("%lu\n", *Counters);
            break;
        }
        for (uintV i = chunk_start; i < chunk_end; i++)
        {
            uintV u = dense_root[i];
            uintV degree = row_ptrs[u + 1] - row_ptrs[u];
            if (degree <= 2048)
                load_neighbor_list(shared_list, cols + row_ptrs[u], degree);
            uintV *hash_table_start = hash_tables + hash_table_sizes[i] * 32;
            uint32_t hash_table_size = hash_table_sizes[i + 1] - hash_table_sizes[i];
            if (hash_table_size <= 128)
            {
                for (uint32_t j = warpIdinBlock; j < hash_table_size; j += blockDim.x / 32)
                {
                    hash_table[j * 32 + threadIdinWarp] = hash_table_start[j * 32 + threadIdinWarp];
                    // if (warpId == 0)
                    // {
                    //     printf("%d\n", hash_table[j * 32 + threadIdinWarp]);
                    // }
                }
                __syncthreads();
                hash_table_start = hash_table;
            }
            for (uint32_t j = warpIdinBlock + 1; j < degree; j += 16)
            {
                uintV v = degree <= 2048? shared_list[j]: cols[row_ptrs[u] + j];
                if (v == 0xFFFFFFFFU)
                    break;
                uintV count = HashIntersectionWithPointer(hash_table_start, hash_table_size, cols + row_ptrs[v], row_ptrs[v + 1] - row_ptrs[v], bin_pointer + warpIdinBlock * 32);
                // uintV count = HashIntersectionCount(shared_list, j, hash_tables + 32 * hash_table_sizes[v], hash_table_sizes[v + 1] - hash_table_sizes[v]);
                // if (warpId == 0)
                // printf("%u %u\n", hash_table_sizes[v + 1] - hash_table_sizes[v], count);
                // uintV count = BinaryIntersectionCountAligned(cols + row_ptrs[v], shared_list, row_ptrs[v + 1] - row_ptrs[v], row_ptrs[u + 1] - row_ptrs[u]);
                if (threadIdinWarp == 0)
                {
                    warpCounter[warpIdinBlock] += count;
                }
                __syncwarp();
            }
            __syncthreads();
        }
    }
}

__global__ void
    __launch_bounds__(128, 16)
        triangle_counting_size512_with_group(
            uintV vertex_count,
            uintE *row_ptrs,
            uintV *cols,
            uintV *root,
            uintV root_num,
            uint64_t *Counters,
            uintV *global_allocator,
            uintV chunk_size)
{
    __shared__ uintV shared_list[512];
    __shared__ uint64_t warpCounter[4];
    __shared__ uintV chunk_start;
    __shared__ uintV chunk_end;
    warpCounter[warpIdinBlock] = 0;
    // coalesced_group active = coalesced_threads();
    // thread_group tile = cooperative_groups:: tiled_partition(this_thread_block(), group_size);
    // uint32_t groupIdinBlock = threadIdx.x / group_size;
    // uint32_t warpIdinGroup = (threadIdx.x % group_size) / 32;
    while (true)
    {
        if (threadIdx.x == 0)
        {
            chunk_start = atomicAdd(global_allocator, chunk_size);
            chunk_end = min(root_num, chunk_start + chunk_size);
        }
        __syncthreads();
        if (chunk_start >= root_num)
        {
            if (threadIdinWarp == 0)
                atomicAdd((unsigned long long *)Counters, (unsigned long long)warpCounter[warpIdinBlock]);
            // printf("%lu\n", *Counters);
            break;
        }
        for (uintV i = chunk_start; i < chunk_end; i++)
        {
            uintV u = root[i];
            uintV degree = row_ptrs[u + 1] - row_ptrs[u];
            load_neighbor_list(shared_list, cols + row_ptrs[u], degree);
            for (uint32_t j = warpIdinBlock + 1; j < degree; j += 4)
            {
                uintV v = shared_list[j];
                if (v == 0xFFFFFFFFU)
                    break;
                // uintV count = HashIntersectionCount(shared_list, j, hash_tables + 32 * hash_table_sizes[v], hash_table_sizes[v + 1] - hash_table_sizes[v]);
                // if (warpId == 0)
                // printf("%u %u\n", hash_table_sizes[v + 1] - hash_table_sizes[v], count);
                uintV count = BinaryIntersectionCountAligned(shared_list, cols + row_ptrs[v], j, row_ptrs[v + 1] - row_ptrs[v]);
                if (threadIdinWarp == 0)
                {
                    warpCounter[warpIdinBlock] += count;
                }
                __syncwarp();
            }
            __syncthreads();
        }
    }
}
__forceinline__ __device__ void load_neighbor_list_async(uintV *dst, const uintV *src, uintV src_size)
{
    if (threadIdx.x < src_size)
    {
        dst[threadIdx.x] = src[threadIdx.x];
    }
}

#define CP_ASYNC_CG(dst, src, Bytes) asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#define CP_ASYNC_WAITALL() asm volatile("cp.async.wait_all;")
__forceinline__ __device__ void load_neighbor_list_async(uintV *dst, const uintV *src, uintV src_size, int thread_num)
{
// #pragma unroll
// for (int i = threadIdx.x; i < src_size; i += thread_num)
// {
//     // dst[i] = src[i];
// }
#pragma unroll
    for (int i = threadIdx.x; i < (src_size + 3) / 4; i += thread_num)
    {
        uint32_t shared_address = __cvta_generic_to_shared(dst + i * 4);
        const uintV *global_address = src + i * 4;
        CP_ASYNC_CG(shared_address, global_address, 16);
    }
}

__forceinline__ __device__ void load_hash_table(uintV* dst, const uintV* hash_table, uintV table_depth)
{
#pragma unroll 
    for (int i = threadIdx.x; i < table_depth * 32; i += blockDim.x)
    {
        dst[i] = hash_table[i];
    }
    __syncthreads();
}

__forceinline__ __device__ void load_hash_table_async(uintV* dst, const uintV* hash_table, uintV table_depth, int thread_num)
{
#pragma unroll
    for (int i = threadIdx.x; i < (table_depth) * 8; i += thread_num)
    {
        uint32_t shared_address = __cvta_generic_to_shared(dst + i * 4);
        const uintV *global_address = hash_table + i * 4;
        CP_ASYNC_CG(shared_address, global_address, 16);
    }
}
__global__ void
    __launch_bounds__(512, 4)
        triangle_counting_with_prefetch(
            uintV vertex_count,
            __restrict__ const uintE *row_ptrs,
            __restrict__ const uintV *cols,
            __restrict__ const uintV *root,
            uintV root_num,
            uint64_t *Counters,
            uintV *global_allocator,
            uintV chunk_size)
{
    __shared__ uintV shared_list[2 * 512];
    __shared__ uint64_t warpCounter[32];
    __shared__ uintV chunk_start;
    __shared__ uintV chunk_end;
    warpCounter[warpIdinBlock] = 0;
    while (true)
    {
        if (threadIdx.x == 0)
        {
            chunk_start = atomicAdd(global_allocator, chunk_size);
            chunk_end = min(root_num, chunk_start + chunk_size);
        }
        __syncthreads();
        if (chunk_start >= root_num)
        {
            if (threadIdinWarp == 0)
                atomicAdd((unsigned long long *)Counters, (unsigned long long)warpCounter[warpIdinBlock]);
            // printf("%lu\n", *Counters);
            break;
        }
        uint32_t read_list = 0;
        // uint8_t write_list = 1;
        load_neighbor_list(shared_list, cols + row_ptrs[root[chunk_start]], row_ptrs[root[chunk_start] + 1] - row_ptrs[root[chunk_start]]);
#pragma unroll
        for (int i = chunk_start; i < chunk_end; i += 1)
        {
            if (i < chunk_end - 1)
            {
                load_neighbor_list_async(shared_list + (1 - read_list) * 512, cols + row_ptrs[root[i + 1]], row_ptrs[root[i + 1] + 1] - row_ptrs[root[i + 1]]);
            }
            for (uint32_t j = warpIdinBlock + 1; j < row_ptrs[root[i] + 1] - row_ptrs[root[i]]; j += 16)
            {
                uintV v = shared_list[read_list * 512 + j];
                if (v == 0xFFFFFFFFU)
                    break;
                uintV count = BinaryIntersectionCountAligned(shared_list + read_list * 512, cols + row_ptrs[v], j, row_ptrs[v + 1] - row_ptrs[v]);
                if (threadIdinWarp == 0)
                    warpCounter[warpIdinBlock] += count;
            }
            read_list = 1 - read_list;
            __syncthreads();
        }
    }
}

__global__ void
    __launch_bounds__(512, 4)
        triangle_counting_with_prefetch1(
            uintV vertex_count,
            __restrict__ const uintE *row_ptrs,
            __restrict__ const uintV *cols,
            __restrict__ const uintV *root,
            uintV root_num,
            uint64_t *Counters,
            uintV *global_allocator,
            uintV chunk_size)
{
    __shared__ __align__(128) uintV shared_list[2 * 512];
    __shared__ uint64_t warpCounter[32];
    __shared__ uintV chunk_start;
    __shared__ uintV chunk_end;
    warpCounter[warpIdinBlock] = 0;
    while (true)
    {
        if (threadIdx.x == 0)
        {
            chunk_start = atomicAdd(global_allocator, chunk_size);
            chunk_end = min(root_num, chunk_start + chunk_size);
        }
        __syncthreads();
        if (chunk_start >= root_num)
        {
            if (threadIdinWarp == 0)
                atomicAdd((unsigned long long *)Counters, (unsigned long long)warpCounter[warpIdinBlock]);
            // printf("%lu\n", *Counters);
            break;
        }
        uint32_t read_list = 0;
        // uint8_t write_list = 1;
        load_neighbor_list(shared_list, cols + row_ptrs[root[chunk_start]], row_ptrs[root[chunk_start] + 1] - row_ptrs[root[chunk_start]]);
        // #pragma unroll
        for (int i = chunk_start; i < chunk_end; i += 1)
        {
            if (i < chunk_end - 1 && threadIdx.x < 128)
            {
                load_neighbor_list_async(shared_list + (1 - read_list) * 512, cols + row_ptrs[root[i + 1]], row_ptrs[root[i + 1] + 1] - row_ptrs[root[i + 1]], 128);
            }
            for (uint32_t j = warpIdinBlock + 1; j < row_ptrs[root[i] + 1] - row_ptrs[root[i]]; j += 16)
            {
                uintV v = shared_list[read_list * 512 + j];
                if (v == 0xFFFFFFFFU)
                    break;
                uintV count = BinaryIntersectionCount(shared_list + read_list * 512, cols + row_ptrs[v], j, row_ptrs[v + 1] - row_ptrs[v]);
                if (threadIdinWarp == 0)
                    warpCounter[warpIdinBlock] += count;
            }
            read_list = 1 - read_list;
            CP_ASYNC_WAITALL();
            __syncthreads();
        }
    }
}

__global__ void
    __launch_bounds__(512, 4)
        triangle_counting_with_hash_prefetch(
            uintV vertex_count,
            __restrict__ const uintE *row_ptrs,
            __restrict__ const uintV *cols,
            __restrict__ const uint32_t *hash_table_sizes,
            __restrict__ const uintV *hash_tables,
            __restrict__ const uintV *root,
            uintV root_num,
            uint64_t *Counters,
            uintV *global_allocator,
            uintV chunk_size)
{
    __shared__ __align__(128) uintV shared_list[2 * 512];
    __shared__ __align__(128) uintV hash_table[32 * 64 * 2];
    __shared__ uint32_t bin_pointer[32 * 16];
    __shared__ uint64_t warpCounter[16];
    __shared__ uintV chunk_start;
    __shared__ uintV chunk_end;
    warpCounter[warpIdinBlock] = 0;
    uint64_t count = 0;
    while (true)
    {
        if (threadIdx.x == 0)
        {
            chunk_start = atomicAdd(global_allocator, chunk_size);
            chunk_end = min(root_num, chunk_start + chunk_size);
        }
        __syncthreads();
        if (chunk_start >= root_num)
        {
            atomicAdd((unsigned long long*) warpCounter + warpIdinBlock, count);
            if (threadIdinWarp == 0)
                atomicAdd((unsigned long long *)Counters, (unsigned long long)warpCounter[warpIdinBlock]);
            // printf("%lu\n", *Counters);
            break;
        }
        load_neighbor_list(shared_list, cols + row_ptrs[root[chunk_start]], row_ptrs[root[chunk_start] + 1] - row_ptrs[root[chunk_start]]);
        load_hash_table(hash_table, hash_tables + hash_table_sizes[chunk_start] * 32, hash_table_sizes[chunk_start + 1] - hash_table_sizes[chunk_start]);
        uint32_t read_list = 0;
        for (uintV i = chunk_start; i < chunk_end; i++)
        {
            uintV u = root[i];
            uintV degree = row_ptrs[u + 1] - row_ptrs[u];
            if (i < chunk_end - 1)
            {
                load_neighbor_list_async(shared_list + (1 - read_list) * 512, cols + row_ptrs[root[i + 1]], row_ptrs[root[i + 1] + 1] - row_ptrs[root[i + 1]], 512);
                load_hash_table_async(hash_table + (1 - read_list)* 32 * 64, hash_tables + hash_table_sizes[i + 1] * 32, hash_table_sizes[i + 2] - hash_table_sizes[i + 1], 512);
            }
// #pragma unroll 4
            for (uint32_t j = warpIdinBlock + 1; j < degree; j+=16)
            {
                uintV v = shared_list[j + read_list * 512];
                // uintV v = cols[row_ptrs[u] + j];
                if (v == 0xFFFFFFFFU)
                    break;
                count += HashIntersectionWithPointer1(hash_table + read_list * 32 * 64, hash_table_sizes[i + 1] - hash_table_sizes[i], cols + row_ptrs[v], row_ptrs[v + 1] - row_ptrs[v], bin_pointer + warpIdinBlock * 32);
            }
            read_list = 1 - read_list;
            CP_ASYNC_WAITALL();
            __syncthreads();
        }
    }
}

