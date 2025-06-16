#include "Meta.h"
#include "Query.h"
#include "utils/context/CudaContext.cuh"
#include "utils/DeviceArray.cuh"
#include "utils/GPUUtil.cuh"
using namespace std;
class ReusePlan
{
    uint32_t** W;
    uint8_t* X;
    uint32_t vertex_num;
    DeviceArray<uint8_t>* dev_reuse_pos_;
    DeviceArray<uint8_t>* dev_reuse_q_cols_;
    Query* query_graph;
public:
    ReusePlan(Query* query);
    uint8_t* GetReusePos(CudaContext* context){
        dev_reuse_pos_ = new DeviceArray<uint8_t>(vertex_num, context);
        HToD(dev_reuse_pos_->GetArray(), X, vertex_num);
        return dev_reuse_pos_->GetArray();
    };
    uint8_t* GetReuseQCols(CudaContext* context){
        size_t vertex_count_ = query_graph->GetVertexCount();
        size_t edge_count_ = query_graph->GetEdgeCount();
        uint8_t* q_cols_ = new uint8_t[edge_count_];
        uint32_t* q_row_ptrs_ = new uint32_t[vertex_count_];
        uint8_t* reuse_q_cols_ = new uint8_t[edge_count_];
        memset(reuse_q_cols_, 0, edge_count_);
        uint8_t writing_pos = 0;
        q_row_ptrs_[0] = 0;
        for (int i = 0; i < vertex_count_; i++)
        {
            for (uintV j = query_graph->GetRowPtrs()[i]; j < query_graph->GetRowPtrs()[i + 1]; j++)
            {
                if (query_graph->GetCols()[j] < i)
                {
                    q_cols_[writing_pos++] = query_graph->GetCols()[j];
                }
            }
            q_row_ptrs_[i + 1] = writing_pos;
        }
        for (int i = 0; i < vertex_count_; i++)
        {
            if (X[i] != 0xFF)
            {
                for (int j = q_row_ptrs_[i]; j < q_row_ptrs_[i + 1]; j++)
                {
                    for (int k = q_row_ptrs_[X[i]]; k < q_row_ptrs_[X[i] + 1]; k++)
                    {
                        if (q_cols_[j] == q_cols_[k]) {reuse_q_cols_[j] = 1; break;}
                    }
                }
            }
        }
        dev_reuse_q_cols_ = new DeviceArray<uint8_t>(edge_count_, context);
        HToD(dev_reuse_q_cols_->GetArray(), reuse_q_cols_, edge_count_);
        delete[] reuse_q_cols_;
        delete[] q_row_ptrs_;
        delete[] q_cols_;
        return dev_reuse_q_cols_->GetArray();
    };
    ~ReusePlan() {
        if (dev_reuse_pos_)
            delete dev_reuse_pos_;
        if (dev_reuse_q_cols_)
            delete dev_reuse_q_cols_;
        for (int i = 0; i < vertex_num; i++)
            delete[] W[i];
        delete[] W;
        delete[] X;
    }
};

ReusePlan::ReusePlan(Query* query)
{
    dev_reuse_pos_ = NULL;
    dev_reuse_q_cols_ = NULL;
    query_graph = query;
    vertex_num = query->GetVertexCount();
    W = new uint32_t* [vertex_num];
    X = new uint8_t[vertex_num];
    for (int i = 0; i < vertex_num; i++) {
        W[i] = new uint32_t[vertex_num];
        memset(W[i], 0, sizeof(uint32_t) * vertex_num);
    }
    vector<set<uint32_t>> backward_neighbors(vertex_num);
    uintE* row_ptrs = query->GetRowPtrs();
    uintV* cols = query->GetCols();
    for (uintV i = 0; i < vertex_num; i++)
    {
        for (uintE j = row_ptrs[i]; j < row_ptrs[i + 1]; j++)
        {
            if (cols[j] < i)
            {
                backward_neighbors[i].insert(cols[j]);
            }
        }
    }
    for (int i = 0; i < vertex_num; i++)
    {
        for (int j = 3; j < i ; j++)
        {
            bool flag = true;
            for (auto n : backward_neighbors[j])
            {
                if (backward_neighbors[i].find(n) == backward_neighbors[i].end())
                {
                    flag = false; break;
                }
            }
            if (flag)
                W[i][j] = backward_neighbors[j].size();
        }
    }
    for (int i = 0; i < vertex_num; i++)
    {
        int reusable = -1;
        uint32_t reuse_value = 0;
        for (int j = 0; j < i; j++)
        {
            if (W[i][j] > reuse_value)
            {
                reuse_value = W[i][j];
                reusable = j;
            }
        }
        X[i] = reusable;
    }
}