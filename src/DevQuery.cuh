#ifndef __GPU_QUERY_CUH__
#define __GPU_QUERY_CUH__
#include "Query.h"
#include "utils/context/CudaContext.cuh"
#include "utils/DeviceArray.cuh"
#include "utils/GPUUtil.cuh"
#include "Meta.h"
class DevQuery: public AbstractGraph
{
public:
    DevQuery(Query* cpu_query, CudaContext* context): context_(context), AbstractGraph(cpu_query->GetDirected())
    {
        BuildDevQuery(cpu_query, context_);
    }
    uint32_t* GetRowPtrs() const{return d_q_row_ptrs_->GetArray();}
    uint8_t* GetCols() const {return d_q_cols_->GetArray();}
    uint8_t* GetRestriction() const {return d_restriction_->GetArray();}
    uint32_t* GetCPURowPtrs() const {return q_row_ptrs_;}
    uint8_t* GetCPUCols() const {return q_cols_;}
    uint8_t* GetCPURestriction() const {return q_restriction_;}
    ~DevQuery() {
    }
protected:
    CudaContext* context_;
    DeviceArray<uint32_t>* d_q_row_ptrs_;
    DeviceArray<uint8_t>* d_q_cols_;
    DeviceArray<uint8_t>* d_restriction_;
    uint32_t* q_row_ptrs_;
    uint8_t* q_cols_;
    uint8_t* q_restriction_;
    void BuildDevQuery(Query* cpu_query, CudaContext* context)
    {
        vertex_count_ = cpu_query->GetVertexCount();
        edge_count_ = cpu_query->GetEdgeCount();
        q_row_ptrs_ = new uint32_t[vertex_count_ + 1];
        q_cols_ = new uint8_t[edge_count_];
        q_restriction_ = new uint8_t[vertex_count_];
        uintE* origin_row_ptrs = cpu_query->GetRowPtrs();
        uintV* origin_cols = cpu_query->GetCols();
        uintV* origin_restriction = cpu_query->GetRestriction();
        q_row_ptrs_[0] = 0;
        uint8_t writing_pos = 0;
        for (int i = 0; i < vertex_count_; i++)
        { 
            q_restriction_[i] = origin_restriction[i];
            for (uintV j = origin_row_ptrs[i]; j < origin_row_ptrs[i + 1]; j++)
            {
                if (origin_cols[j] < i)
                    q_cols_[writing_pos++] = origin_cols[j];
            }
            q_row_ptrs_[i + 1] = writing_pos;
        }
        cudaStream_t stream = context -> Stream();
        d_q_row_ptrs_ = new DeviceArray<uint32_t> (vertex_count_ + 1, context);
        d_q_cols_ = new DeviceArray<uint8_t> (writing_pos, context);
        d_restriction_ = new DeviceArray<uint8_t> (vertex_count_,  context);
        CUDA_ERROR(cudaStreamSynchronize(stream));
        HToD<uint32_t>(d_q_row_ptrs_->GetArray(), q_row_ptrs_, vertex_count_ + 1, stream);
        HToD<uint8_t>(d_q_cols_->GetArray(), q_cols_, writing_pos, stream);
        HToD<uint8_t>(d_restriction_->GetArray(), q_restriction_, vertex_count_, stream);
        CUDA_ERROR(cudaStreamSynchronize(stream));
    }
};
#endif