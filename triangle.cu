#include "src/include.cuh"
#include "utils/CommandLine.h"
#include <stdio.h>
#include <x86intrin.h>
using namespace std;
#ifndef DEVICEIDX
#define DEVICEIDX dev_id
#endif

#define DENSE_VERTEX_THRESH 16

CudaContextManager *CudaContextManager::gCudaContextManager = NULL;
__inline__ uint64_t perf_counter(void)
{
    __asm__ __volatile__("" : : : "memory");
    uint64_t r = __rdtsc();
    __asm__ __volatile__("" : : : "memory");

    return r;
}

int main(int argc, char *argv[])
{
    CommandLine cmd(argc, argv);
    std::string data_filename = cmd.GetOptionValue("-f", "../data/com-dblp.ungraph.txt");
    int dense_threshold = cmd.GetOptionIntValue("-dense", 256);
    int middle_threshold = cmd.GetOptionIntValue("-middle", 128);
    int algorithm = cmd.GetOptionIntValue("-algo", 1);
    int dense_chunk = cmd.GetOptionIntValue("-dc", 32);
    int sparse_chunk = cmd.GetOptionIntValue("-sc", 32);
    int middle_chunk = cmd.GetOptionIntValue("-mc", 32);
    int group_size = cmd.GetOptionIntValue("-gs", 64);
    int dev_id = 0;
    data_filename = data_filename;
    Graph data_graph(data_filename, false);
    // data_graph.TrianglePeel();
    data_graph.TriangleReorder();
    data_graph.TriangleOrientation();
    cout << data_graph.GetMaxDegree() << endl;
    CudaContextType cuda_context_type = CNMEM_MANAGED;
    CudaContextManager::CreateCudaContextManager(1, cuda_context_type);
    auto context = CudaContextManager::GetCudaContextManager()->GetCudaContext(DEVICEIDX);
    CUDA_ERROR(cudaSetDevice(context->GetDeviceMemoryInfo()->GetDevId()));
    DevGraph *gpu_graph = new DevGraph(&data_graph, context);
    AlignedDevGraph *gpu_aligned_graph = new AlignedDevGraph(&data_graph, context);
    // uint8_t init_layer = 2;
    // uintE init_num;
    // uintE init_task_num;
    DeviceArray<uintV> *global_allocator = new DeviceArray<uintV>(1, context);
    DeviceArray<uintV> *global_allocator_middle = new DeviceArray<uintV>(1, context);
    DeviceArray<uintV> *global_allocator_sparse = new DeviceArray<uintV>(1, context);
    DeviceArray<uint64_t> *Counter = new DeviceArray<uint64_t>(1, context);
    CUDA_ERROR(cudaMemset(global_allocator->GetArray(), 0, sizeof(uintE)));
    CUDA_ERROR(cudaMemset(global_allocator_middle->GetArray(), 0, sizeof(uintE)));
    CUDA_ERROR(cudaMemset(global_allocator_sparse->GetArray(), 0, sizeof(uintE)));
    CUDA_ERROR(cudaMemset(Counter->GetArray(), 0, sizeof(uint64_t)));
    vector<uintV> dense_vertex;
    vector<uintV> middle_vertex;
    vector<uintV> sparse_vertex;
    // uintV dense_threshold = 64;
    for (uintV i = 0; i < data_graph.GetVertexCount(); i++)
    {
        if (data_graph.GetRowPtrs()[i + 1] - data_graph.GetRowPtrs()[i] >= dense_threshold)
        {
            dense_vertex.push_back(i);
        }
        else if (data_graph.GetRowPtrs()[i + 1] - data_graph.GetRowPtrs()[i] >= middle_threshold)
            middle_vertex.push_back(i);
        else
            sparse_vertex.push_back(i);
    }
    DeviceArray<uintV> *dense_vertices = new DeviceArray<uintV>(dense_vertex.size(), context);
    HToD<uintV>(dense_vertices->GetArray(), dense_vertex.data(), dense_vertex.size());
    DeviceArray<uintV> *sparse_vertices = new DeviceArray<uintV>(sparse_vertex.size(), context);
    HToD<uintV>(sparse_vertices->GetArray(), sparse_vertex.data(), sparse_vertex.size());
    DeviceArray<uintV> *middle_vertices = new DeviceArray<uintV>(middle_vertex.size(), context);
    HToD<uintV>(middle_vertices->GetArray(), middle_vertex.data(), middle_vertex.size());
    cout << dense_vertex.size() << endl;
    cout << middle_vertex.size() << endl;
    cout << sparse_vertex.size() << endl;
    CuckooGraph* cuckoograph;
    if (algorithm == 4) cuckoograph = new CuckooGraph(&data_graph, context, dense_vertex.data(), dense_vertex.size());
    HashGraph* hashgraph;
    HashGraph* dense_hashgraph;
    HashGraph* middle_hashgraph;
    if (algorithm == 5) hashgraph = new HashGraph(&data_graph, context);
    if (algorithm == 6) hashgraph = new HashGraph(&data_graph, dense_vertex.data(), dense_vertex.size(), context);
    if (algorithm == 10) {
        dense_hashgraph = new HashGraph(&data_graph, dense_vertex.data(), dense_vertex.size(), context);
        middle_hashgraph = new HashGraph(&data_graph, middle_vertex.data(), middle_vertex.size(), context);
    }
    GPUTimer timer;
    cudaStream_t stream = CudaContextManager::GetCudaContextManager()->GetCudaContext(DEVICEIDX)->Stream();
    CUDA_ERROR(cudaStreamSynchronize(stream));
    timer.StartTimer();
    if (algorithm == 0)
        triangle_counting<<<216, 1024, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_graph->GetRowPtrs(),
            gpu_graph->GetCols(),
            Counter->GetArray(),
            global_allocator->GetArray(),
            sparse_chunk); // on Graph500-24: ~3000ms
    else if (algorithm == 1)
    {
        triangle_counting_with_shared_memory<<<432, 512, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_aligned_graph->GetRowPtrs(),
            gpu_aligned_graph->GetCols(),
            dense_vertices->GetArray(),
            dense_vertex.size(),
            Counter->GetArray(),
            global_allocator->GetArray(),
            dense_chunk); // on Graph500-24: ~1350ms
        // triangle_counting_sparse_with_subwarp<<<216, 1024, 0, stream>>>(
        //     data_graph.GetVertexCount(),
        //     gpu_aligned_graph->GetRowPtrs(),
        //     gpu_aligned_graph->GetCols(),
        //     sparse_vertices->GetArray(),
        //     sparse_vertex.size(),
        //     Counter->GetArray(),
        //     global_allocator_sparse->GetArray(),
        //     sparse_chunk);
    }
    else if (algorithm == 2)
    {
        triangle_counting_with_shared_memory<<<432, 512, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_graph->GetRowPtrs(),
            gpu_graph->GetCols(),
            dense_vertices->GetArray(),
            dense_vertex.size(),
            Counter->GetArray(),
            global_allocator->GetArray(),
            dense_chunk); // on Graph500-24: ~1350ms

        triangle_counting_sparse_with_subwarp<<<216, 1024, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_aligned_graph->GetRowPtrs(),
            gpu_aligned_graph->GetCols(),
            sparse_vertices->GetArray(),
            sparse_vertex.size(),
            Counter->GetArray(),
            global_allocator_sparse->GetArray(),
            sparse_chunk);
    }
    else if (algorithm == 3)
    {
        triangle_counting_with_shared_memory_unroll<<<432, 512, 0, stream>>>(
            data_graph.GetVertexCount(),
            // gpu_aligned_graph->GetRowPtrs(),
            // gpu_aligned_graph->GetCols(),
            gpu_graph->GetRowPtrs(),
            gpu_graph->GetCols(),
            dense_vertices->GetArray(),
            dense_vertex.size(),
            Counter->GetArray(),
            global_allocator->GetArray(),
            dense_chunk); // on Graph500-24: ~1350ms
        triangle_counting_sparse_with_subwarp<<<216, 1024, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_aligned_graph->GetRowPtrs(),
            gpu_aligned_graph->GetCols(),
            sparse_vertices->GetArray(),
            sparse_vertex.size(),
            Counter->GetArray(),
            global_allocator_sparse->GetArray(),
            sparse_chunk);
    }
    else if (algorithm == 4)
    {
        triangle_counting_with_shared_memory_cuckoo<<<432, 512, 0, stream>>>(
            data_graph.GetVertexCount(),
            // gpu_aligned_graph->GetRowPtrs(),
            // gpu_aligned_graph->GetCols(),
            gpu_graph->GetRowPtrs(),
            gpu_graph->GetCols(),
            dense_vertices->GetArray(),
            dense_vertex.size(),
            cuckoograph->GetHashTableSizes(),
            cuckoograph->GetHashTables(),
            Counter->GetArray(),
            global_allocator->GetArray(),
            dense_chunk); // on Graph500-24: ~1350ms
        triangle_counting_sparse_with_subwarp<<<216, 1024, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_aligned_graph->GetRowPtrs(),
            gpu_aligned_graph->GetCols(),
            sparse_vertices->GetArray(),
            sparse_vertex.size(),
            Counter->GetArray(),
            global_allocator_sparse->GetArray(),
            sparse_chunk);
    }
    else if (algorithm == 5)
    {
        triangle_counting_with_hash<<<432, 512, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_graph->GetRowPtrs(),
            gpu_graph->GetCols(),
            dense_vertices->GetArray(),
            dense_vertex.size(),
            hashgraph->GetTableSizes(),
            hashgraph->GetHashTables(),
            Counter->GetArray(),
            global_allocator->GetArray(),
            dense_chunk
        );
        triangle_counting_sparse_with_subwarp<<<216, 1024, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_aligned_graph->GetRowPtrs(),
            gpu_aligned_graph->GetCols(),
            sparse_vertices->GetArray(),
            sparse_vertex.size(),
            Counter->GetArray(),
            global_allocator_sparse->GetArray(),
            sparse_chunk);
    }
    else if (algorithm == 6)
    {
        triangle_counting_with_hash1<<<432, 512, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_graph->GetRowPtrs(),
            gpu_graph->GetCols(),
            dense_vertices->GetArray(),
            dense_vertex.size(),
            hashgraph->GetTableSizes(),
            hashgraph->GetHashTables(),
            Counter->GetArray(),
            global_allocator->GetArray(),
            dense_chunk
        );
        // triangle_counting_sparse_with_subwarp<<<216, 1024, 0, stream>>>(
        //     data_graph.GetVertexCount(),
        //     gpu_aligned_graph->GetRowPtrs(),
        //     gpu_aligned_graph->GetCols(),
        //     sparse_vertices->GetArray(),
        //     sparse_vertex.size(),
        //     Counter->GetArray(),
        //     global_allocator_sparse->GetArray(),
        //     sparse_chunk);
    }
    else if (algorithm == 7)
    {
        triangle_counting_size512_with_group<<<1728, 128, 0, stream>>> (
            data_graph.GetVertexCount(),
            gpu_graph->GetRowPtrs(),
            gpu_graph->GetCols(),
            middle_vertices->GetArray(),
            middle_vertex.size(),
            Counter->GetArray(),
            global_allocator->GetArray(),
            middle_chunk
        );
    }
    else if (algorithm == 8) {
        triangle_counting_with_prefetch<<<432, 512, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_graph->GetRowPtrs(),
            gpu_graph->GetCols(),
            middle_vertices->GetArray(),
            middle_vertex.size(),
            Counter->GetArray(),
            global_allocator->GetArray(),
            middle_chunk
        );
    }
    else if (algorithm == 9) {
        triangle_counting_with_prefetch1<<<432, 512, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_aligned_graph->GetRowPtrs(),
            gpu_aligned_graph->GetCols(),
            middle_vertices->GetArray(),
            middle_vertex.size(),
            Counter->GetArray(),
            global_allocator->GetArray(),
            middle_chunk
        );
    }
    else if (algorithm == 10) {
        triangle_counting_with_hash1<<<432, 512, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_aligned_graph->GetRowPtrs(),
            gpu_aligned_graph->GetCols(),
            dense_vertices->GetArray(),
            dense_vertex.size(),
            dense_hashgraph->GetTableSizes(),
            dense_hashgraph->GetHashTables(),
            Counter->GetArray(),
            global_allocator->GetArray(),
            dense_chunk
        );
        triangle_counting_with_hash_prefetch<<<432, 512, 0, stream>>> (
            data_graph.GetVertexCount(),
            gpu_aligned_graph->GetRowPtrs(),
            gpu_aligned_graph->GetCols(),
            middle_hashgraph->GetTableSizes(),
            middle_hashgraph->GetHashTables(),
            middle_vertices->GetArray(),
            middle_vertex.size(),
            Counter->GetArray(),
            global_allocator_middle->GetArray(),
            middle_chunk
        );
        triangle_counting_with_shared_memory<<<432, 512, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_aligned_graph->GetRowPtrs(),
            gpu_aligned_graph->GetCols(),
            sparse_vertices->GetArray(),
            sparse_vertex.size(),
            Counter->GetArray(),
            global_allocator_sparse->GetArray(),
            sparse_chunk);
        // triangle_counting_with_prefetch1<<<432, 512, 0, stream>>>(
        //     data_graph.GetVertexCount(),
        //     gpu_aligned_graph->GetRowPtrs(),
        //     gpu_aligned_graph->GetCols(),
        //     sparse_vertices->GetArray(),
        //     sparse_vertex.size(),
        //     Counter->GetArray(),
        //     global_allocator_sparse->GetArray(),
        //     sparse_chunk
        // );
    }
    CUDA_ERROR(cudaStreamSynchronize(stream));
    timer.EndTimer();
    uint64_t counts;
    DToH<uint64_t>(&counts, Counter->GetArray(), 1);
    CUDA_ERROR(cudaStreamSynchronize(stream));
    cout << counts << endl;
    cout << timer.GetElapsedMilliSeconds() << "ms" << endl;
    // CPUMatch match(&data_graph, &query_graph);
    // auto startclock = clock();
    // uint64_t cpu_counts = match.count();
    // auto endclock = clock();
    // cout << match.count() << endl;
    // cout << (endclock - startclock) / (CLOCKS_PER_SEC / 1000) << "ms" << endl;
}