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


class DevReuseQuery: public AbstractGraph
{
public:
    DevReuseQuery(ReuseQuery* cpu_query, CudaContext* context): context_(context), AbstractGraph(cpu_query->GetDirected())
    {
        BuildDevQuery(cpu_query, context_);
    }
    uint32_t* GetRowPtrs() const{return d_q_row_ptrs_->GetArray();}
    uint8_t* GetCols() const {return d_q_cols_->GetArray();}
    uint8_t* GetRestriction() const {return d_restriction_->GetArray();}
    uint32_t* GetCPURowPtrs() const {return q_row_ptrs_;}
    uint8_t* GetCPUCols() const {return q_cols_;}
    uint8_t* GetCPURestriction() const {return q_restriction_;}
    uint8_t* GetCPUReuse() const {return q_reuse_;}
    uint8_t* GetCPUIsReused() const {return q_is_reused_;}
    ~DevReuseQuery() {
    }
protected:
    CudaContext* context_;
    DeviceArray<uint32_t>* d_q_row_ptrs_;
    DeviceArray<uint8_t>* d_q_cols_;
    DeviceArray<uint8_t>* d_restriction_;
    DeviceArray<uint8_t>* d_q_reuse_;
    uint32_t* q_row_ptrs_;
    uint8_t* q_cols_;
    uint8_t* q_restriction_;
    uint8_t* q_reuse_;
    uint8_t* q_is_reused_;
    void BuildDevQuery(ReuseQuery* cpu_query, CudaContext* context)
    {
        vertex_count_ = cpu_query->GetVertexCount();
        edge_count_ = cpu_query->GetEdgeCount();
        q_row_ptrs_ = new uint32_t[vertex_count_ + 1];
        q_cols_ = new uint8_t[edge_count_];
        q_restriction_ = new uint8_t[vertex_count_];
        q_reuse_ = new uint8_t[vertex_count_];
        q_is_reused_ = new uint8_t[vertex_count_];
        uintE* origin_row_ptrs = cpu_query->GetRowPtrs();
        uintV* origin_cols = cpu_query->GetCols();
        uintV* origin_restriction = cpu_query->GetRestriction();
        uintV* origin_reuse = cpu_query->GetReuse();
        bool* origin_is_reused = cpu_query->GetIsReused();
        q_row_ptrs_[0] = 0;
        uint8_t writing_pos = 0;
        for (int i = 0; i < vertex_count_; i++)
        { 
            q_restriction_[i] = origin_restriction[i];
            q_reuse_[i] = origin_reuse[i];
            q_is_reused_[i] = origin_is_reused[i];
            for (uintV j = origin_row_ptrs[i]; j < origin_row_ptrs[i + 1]; j++)
            {
                if (origin_cols[j] < i)
                    q_cols_[writing_pos++] = origin_cols[j];
            }
            q_row_ptrs_[i + 1] = writing_pos;
        }
        for (int i = 0; i < vertex_count_; i++)
        {
            if (q_reuse_[i] != 0xFF)
            {
                for (uint8_t j = q_row_ptrs_[q_reuse_[i]]; j < q_row_ptrs_[q_reuse_[i] + 1]; j++)
                {
                    uint8_t v = q_cols_[j] & 0x7f;
                    for (uint8_t k = q_row_ptrs_[i]; k < q_row_ptrs_[i + 1]; k++)
                    {
                        if ((q_cols_[k] & 0x7f) == v) {
                            q_cols_[k] |= 0x80;
                        }
                    }
                }
            }
        }
        cudaStream_t stream = context -> Stream();
        d_q_row_ptrs_ = new DeviceArray<uint32_t> (vertex_count_ + 1, context);
        d_q_cols_ = new DeviceArray<uint8_t> (writing_pos, context);
        d_restriction_ = new DeviceArray<uint8_t> (vertex_count_,  context);
        d_q_reuse_ = new DeviceArray<uint8_t> (vertex_count_, context);
        CUDA_ERROR(cudaStreamSynchronize(stream));
        HToD<uint32_t>(d_q_row_ptrs_->GetArray(), q_row_ptrs_, vertex_count_ + 1, stream);
        HToD<uint8_t>(d_q_cols_->GetArray(), q_cols_, writing_pos, stream);
        HToD<uint8_t>(d_restriction_->GetArray(), q_restriction_, vertex_count_, stream);
        HToD<uint8_t>(d_q_reuse_->GetArray(), q_reuse_, vertex_count_, stream);
        CUDA_ERROR(cudaStreamSynchronize(stream));
    }
};
#endif