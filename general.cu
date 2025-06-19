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
    std::string query_filename = cmd.GetOptionValue("-q", "../queries/Q1.txt");
    int algorithm = cmd.GetOptionIntValue("-algo", 1);
    int dev_id = 0;
    data_filename = data_filename;
    Graph data_graph(data_filename, false);
    Query query_graph(query_filename);
    cout << data_graph.GetMaxDegree() << endl;
    CudaContextType cuda_context_type = CNMEM_MANAGED;
    CudaContextManager::CreateCudaContextManager(1, cuda_context_type);
    auto context = CudaContextManager::GetCudaContextManager()->GetCudaContext(DEVICEIDX);
    CUDA_ERROR(cudaSetDevice(context->GetDeviceMemoryInfo()->GetDevId()));
    DevGraph *gpu_graph = new DevGraph(&data_graph, context);
    DevQuery *gpu_query = new DevQuery(&query_graph, context);
    uint8_t init_layer = 2;
    // uintE init_num;
    bool init_oriented = (query_graph.GetRestriction()[1] != 0xFF);
    cout << init_oriented << endl;
    uintE init_task_num = init_oriented? data_graph.GetEdgeCount() / 2 : data_graph.GetEdgeCount();
    DeviceArray<uint64_t> *global_allocator = new DeviceArray<uint64_t>(1, context);
    DeviceArray<uint64_t> *Counter = new DeviceArray<uint64_t>(1, context);
    DeviceArray<uint32_t> *IdleWarpCounter = new DeviceArray<uint32_t>(1, context);
    DeviceArray<uintV> *BFSlayer = new DeviceArray<uintV>(216 * (data_graph.GetMaxDegree() + 1023) / 1024 * 1024, context);
    DeviceArray<uintV> *init_match = new DeviceArray<uintV>(init_task_num * init_layer, context);
    uintV* cpu_init_match = new uintV[init_task_num * init_layer];
    uintE curr_pos = 0;
    for (uintV i = 0; i < data_graph.GetVertexCount(); i++)
    {
        for (uintE j = data_graph.GetRowPtrs()[i]; j < data_graph.GetRowPtrs()[i + 1]; j++)
        {
            if (init_oriented && data_graph.GetCols()[j] > i) continue;
            cpu_init_match[curr_pos * 2] = i;
            cpu_init_match[curr_pos * 2 + 1] = data_graph.GetCols()[j];
            curr_pos++;
        }
    }
    HToD<uintV>(init_match->GetArray(), cpu_init_match, init_task_num * init_layer);
    CUDA_ERROR(cudaMemset(global_allocator->GetArray(), 0, sizeof(uint64_t)));
    CUDA_ERROR(cudaMemset(Counter->GetArray(), 0, sizeof(uint64_t)));
    CUDA_ERROR(cudaMemset(IdleWarpCounter->GetArray(), 0, sizeof(uint32_t)));

    Stack stack(WARP_PER_BLOCK * 216, data_graph.GetMaxDegree(), query_graph.GetVertexCount(), context);
    WorkerQueue *queue = init_device_WorkerQueue(216 * 32, context);
    TaskQueue* task_queue = new TaskQueue;
    int max_queue_size = 65536 * 1024;
    task_queue->queue_size = max_queue_size;
    task_queue->head = 0;
    task_queue->tail = 0;
    task_queue->task_count = 0;
    DeviceArray<Task1>* tasks = new DeviceArray<Task1>(max_queue_size, context);
    CUDA_ERROR(cudaMemset(tasks->GetArray(), 0, sizeof(Task) * max_queue_size));
    task_queue->tasks = tasks->GetArray();
    DeviceArray<TaskQueue>* gpu_task_queue = new DeviceArray<TaskQueue>(1, context);
    HToD<TaskQueue>(gpu_task_queue->GetArray(), task_queue, 1);
    MatchInfo cpu_info;
    cpu_info.row_ptrs = gpu_graph->GetRowPtrs();
    cpu_info.cols = gpu_graph->GetCols();
    cpu_info.stacks = stack.allocate_stacks();
    cpu_info.init_tasks = init_match->GetArray();
    cpu_info.init_task_num = init_task_num;
    cpu_info.dg_vertex_count = data_graph.GetVertexCount();
    cpu_info.stack_layer_size = (data_graph.GetMaxDegree() + 31) / 32 * 32,
    cpu_info.q_vertex_count = query_graph.GetVertexCount();
    cpu_info.init_layer = init_layer;
    cpu_info.q_row_ptrs[0] = 0;
    cpu_info.task_queue = gpu_task_queue->GetArray();
    for (uint8_t i = 0; i < cpu_info.q_vertex_count; i++)
    {
        cpu_info.q_row_ptrs[i + 1] = gpu_query->GetCPURowPtrs()[i + 1];
        cpu_info.q_restriction[i] = gpu_query->GetCPURestriction()[i];
        for (uint8_t j = cpu_info.q_row_ptrs[i]; j < cpu_info.q_row_ptrs[i + 1]; j ++ )
        {
            cpu_info.q_cols[j] = gpu_query->GetCPUCols()[j];
        }
    }
    DeviceArray<MatchInfo>* gpu_match_info = new DeviceArray<MatchInfo>(1, context);
    HToD<MatchInfo>(gpu_match_info->GetArray(), &cpu_info, 1);
    DeviceArray<uint64_t>* profileCounter = new DeviceArray<uint64_t>(216 * 32, context);
    GPUTimer timer;
    cudaStream_t stream = CudaContextManager::GetCudaContextManager()->GetCudaContext(DEVICEIDX)->Stream();
    CUDA_ERROR(cudaStreamSynchronize(stream));
    cout << "start searching" << endl;
    timer.StartTimer();
    if (algorithm == 0)
    {
        EdgeCentricDFSBalancedKernel<<<216, 1024, 0, stream>>>(
            data_graph.GetVertexCount(),
            gpu_graph->GetRowPtrs(),
            gpu_graph->GetCols(),
            (data_graph.GetMaxDegree() + 31) / 32 * 32,
            BFSlayer->GetArray(),
            cpu_info.stacks,
            query_graph.GetVertexCount(),
            gpu_query->GetRowPtrs(),
            gpu_query->GetCols(),
            gpu_query->GetRestriction(),
            init_task_num,
            init_match->GetArray(),
            Counter->GetArray(),
            3,
            queue
        );
    }
    else if (algorithm == 1)
    {
        DFSWarpKernel<<<216, 1024, 0, stream>>>(
            gpu_match_info->GetArray(),
            global_allocator->GetArray(),
            Counter->GetArray(),
            profileCounter->GetArray()
        );
    }
    else if (algorithm == 2)
    {
        DFSWarpBalanceKernel<<<108, 1024, 0, stream>>>(
            gpu_match_info->GetArray(),
            global_allocator->GetArray(),
            Counter->GetArray(),
            profileCounter->GetArray(),
            IdleWarpCounter->GetArray()
        );
    }
    CUDA_ERROR(cudaStreamSynchronize(stream));
    timer.EndTimer();
    uint64_t counts;
    DToH<uint64_t>(&counts, Counter->GetArray(), 1);
    CUDA_ERROR(cudaStreamSynchronize(stream));
    // profileCounter->Print();
    uint32_t warpN = 216 * 32;
    if (algorithm == 2) warpN = 108 * 32;
    uint64_t* cpu_profileCounter = new uint64_t[warpN];
    DToH<uint64_t>(cpu_profileCounter, profileCounter->GetArray(), warpN);
    sort(cpu_profileCounter, cpu_profileCounter + warpN);
    for (int i = 0; i < warpN; i ++)
    {
        if (i % 20 == 0) cout << endl;
        cout << cpu_profileCounter[i] << " ";
    }
    cout << endl;
    cout << counts << endl;
    cout << timer.GetElapsedMilliSeconds() << "ms" << endl;
    // CPUMatch match(&data_graph, &query_graph);
    // auto startclock = clock();
    // uint64_t cpu_counts = match.count();
    // auto endclock = clock();
    // cout << match.count() << endl;
    // cout << (endclock - startclock) / (CLOCKS_PER_SEC / 1000) << "ms" << endl;
}