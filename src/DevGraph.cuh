#ifndef __GPU_DEV_GRAPH_CUH__
#define __GPU_DEV_GRAPH_CUH__

#include "AbstractGraph.h"
#include "CPUGraph.h"
#include "utils/context/CudaContext.cuh"
#include "utils/DeviceArray.cuh"
#include "utils/GPUUtil.cuh"
#include "Meta.h"

class DevGraph : public AbstractGraph
{
public:
    DevGraph(Graph *cpu_graph, CudaContext *context) : AbstractGraph(cpu_graph->GetDirected()), context_(context)
    {
        BuildDevGraph(cpu_graph, context_);
    }

    uintE* GetRowPtrs()
    {
        return d_row_ptrs_->GetArray();
    }
    uintV* GetCols()
    {
        return d_cols_->GetArray();
    }
protected:
    CudaContext *context_;
    DeviceArray<uintE> *d_row_ptrs_;
    DeviceArray<uintV> *d_cols_;
    void BuildDevGraph(Graph *cpu_graph, CudaContext *context)
    {
        cudaStream_t stream = context->Stream();
        vertex_count_ = cpu_graph->GetVertexCount();
        edge_count_ = cpu_graph->GetEdgeCount();
        d_row_ptrs_ = new DeviceArray<uintE>(vertex_count_ + 1, context);
        d_cols_ = new DeviceArray<uintV>(edge_count_, context);
        CUDA_ERROR(cudaMemcpyAsync(d_row_ptrs_->GetArray(), cpu_graph->GetRowPtrs(),
                                   sizeof(uintE) * (vertex_count_ + 1),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_ERROR(cudaMemcpyAsync(d_cols_->GetArray(), cpu_graph->GetCols(),
                                   sizeof(uintV) * edge_count_,
                                   cudaMemcpyHostToDevice, stream));
    }
};

#endif