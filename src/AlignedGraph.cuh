#ifndef __GPU_ALIGNED_GRAPH_CUH__
#define __GPU_ALIGNED_GRAPH_CUH__

#include "AbstractGraph.h"
#include "CPUGraph.h"
#include "utils/context/CudaContext.cuh"
#include "utils/DeviceArray.cuh"
#include "utils/GPUUtil.cuh"
#include "Meta.h"

class AlignedDevGraph : public AbstractGraph
{
public:
    AlignedDevGraph(Graph *cpu_graph, CudaContext *context) : AbstractGraph(cpu_graph->GetDirected()), context_(context)
    {
        d_row_ptrs_ = NULL;
        d_cols_ = NULL;
        BuildDevGraph(cpu_graph, context_);
    }

    uintE* GetRowPtrs()
    {
        return d_row_ptrs_->GetArray();
        // return d_row_ptrs_;
    }
    uintV* GetCols()
    {
        return d_cols_->GetArray();
        // return d_cols_;
    }
protected:
    CudaContext *context_;
    DeviceArray<uintE> *d_row_ptrs_;
    DeviceArray<uintV> *d_cols_;
    // uintE* d_row_ptrs_;
    // uintV* d_cols_;
    void BuildDevGraph(Graph *cpu_graph, CudaContext *context)
    {
        cudaStream_t stream = context->Stream();
        vertex_count_ = cpu_graph->GetVertexCount();
        edge_count_ = cpu_graph->GetEdgeCount();
        uintE* aligned_row_ptrs = new uintE[vertex_count_  + 1];
        uintE current_pos = cpu_graph->GetRowPtrs()[1];
        for (int i = 1; i < vertex_count_; i++)
        {
            uint32_t degree = cpu_graph->GetRowPtrs()[i + 1] - cpu_graph->GetRowPtrs()[i];
            if ((current_pos + degree) / 32  - (current_pos) / 32)
            {
                current_pos = (current_pos + 31) / 32 * 32;
            }
            aligned_row_ptrs[i] = current_pos;
            current_pos += degree;
        }
        aligned_row_ptrs[vertex_count_] = current_pos;
        uintV* aligned_cols = new uintV[current_pos];
        memset(aligned_cols, -1, sizeof(uintV) * current_pos);
        for (uintV i = 0; i < vertex_count_; i++)
        {
            for (uintE j = 0; j < cpu_graph->GetRowPtrs()[i + 1] - cpu_graph->GetRowPtrs()[i]; j++)
            {
                aligned_cols[aligned_row_ptrs[i] + j] = cpu_graph->GetCols()[cpu_graph->GetRowPtrs()[i] + j];
            }
        }
        CUDA_ERROR(cudaMalloc(&d_row_ptrs_, (vertex_count_ + 1) * sizeof(uintE)));
        CUDA_ERROR(cudaMalloc(&d_cols_, current_pos * sizeof(uintV)));
        d_row_ptrs_ = new DeviceArray<uintE>(vertex_count_ + 1, context);
        d_cols_ = new DeviceArray<uintV>(current_pos, context);
        cout << d_cols_->GetArray() << endl;
        // cout << d_row_ptrs_ << endl;
        // cout << d_cols_ << endl;
        CUDA_ERROR(cudaMemcpyAsync(d_row_ptrs_->GetArray(), aligned_row_ptrs,
                                   sizeof(uintE) * (vertex_count_ + 1),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_ERROR(cudaMemcpyAsync(d_cols_->GetArray(), aligned_cols,
                                   sizeof(uintV) * current_pos,
                                   cudaMemcpyHostToDevice, stream));
        // CUDA_ERROR(cudaMemcpyAsync(d_row_ptrs_, aligned_row_ptrs,
        //                            sizeof(uintE) * (vertex_count_ + 1),
        //                            cudaMemcpyHostToDevice, stream));
        // CUDA_ERROR(cudaMemcpyAsync(d_cols_, aligned_cols,
        //                            sizeof(uintV) * current_pos,
        //                            cudaMemcpyHostToDevice, stream));
    }
};

#endif