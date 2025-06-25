#ifndef _HASH_GRAPH_H
#define _HASH_GRAPH_H
#include "AbstractGraph.h"
#include "CPUGraph.h"
#include "utils/context/CudaContext.cuh"
#include "utils/DeviceArray.cuh"
#include "utils/GPUUtil.cuh"
#include "Meta.h"
#include "Hash.cuh"

class HashTable
{
public:
    size_t table_size;
    vector<vector<uintV>> table;
    HashTable(uintV *list, uintV list_size)
    {
        table_size = 0;
        table.resize(32);
        for (auto bucket : table)
        {
            bucket.clear();
        }
        for (uintV i = 0; i < list_size; i++)
        {
            uintV v = list[i];
            uint32_t bucket = hash1(v) % 32;
            table[bucket].push_back(v);
            table_size = max(table_size, table[bucket].size());
        }
    }
    HashTable(uintV *list, uintV list_size, uint32_t bin_num)
    {
        table_size = 0;
        table.resize(bin_num);
        for (auto bucket : table)
        {
            bucket.clear();
        }
        for (uintV i = 0; i < list_size; i++)
        {
            uintV v = list[i];
            uint32_t bucket = hash1(v) % bin_num;
            table[bucket].push_back(v);
            table_size++;
        }
    }
};

class HashGraph
{
private:
    CudaContext *context_;
    uint32_t *table_sizes;
    uintV *hash_tables;
    uintV vertex_num;
    DeviceArray<uint32_t> *d_table_sizes;
    DeviceArray<uintV> *d_hash_tables;

public:
    HashGraph(Graph *cpu_graph, uintV *dense_vertex, uintV dense_vertex_num, CudaContext *context)
    {
        context_ = context;
        vertex_num = cpu_graph->GetVertexCount();
        table_sizes = NULL;
        hash_tables = NULL;
        d_table_sizes = NULL;
        d_hash_tables = NULL;
        uintV *cols = cpu_graph->GetCols();
        uintE *row_ptrs = cpu_graph->GetRowPtrs();
        vector<HashTable *> tables;
        table_sizes = new uint32_t[dense_vertex_num + 1];
        table_sizes[0] = 0;
        uint32_t table_size_counts[128] = {0};
        for (uintV i = 0; i < dense_vertex_num; i++)
        {
            HashTable *new_table = new HashTable(cols + row_ptrs[dense_vertex[i]], row_ptrs[dense_vertex[i] + 1] - row_ptrs[dense_vertex[i]]);
            tables.push_back(new_table);
            table_sizes[i + 1] = table_sizes[i] + new_table->table_size;
            table_size_counts[new_table->table_size] += 1;
        }
        // for (int i = 0; i < 128; i++)
        //     cout << "table_size: "<< i << "   vertex_num: " << table_size_counts[i] << endl;
        hash_tables = new uintV[table_sizes[dense_vertex_num] * 32];
        printf("hash_table_total_size: %d\n", table_sizes[dense_vertex_num]);
        memset(hash_tables, 0xFF, sizeof(uintV) * table_sizes[dense_vertex_num] * 32);
        for (uintV i = 0; i < dense_vertex_num; i++)
        {
            for (int bin_id = 0; bin_id < 32; bin_id++)
            {
                for (int j = 0; j < tables[i]->table[bin_id].size(); j++)
                {
                    hash_tables[table_sizes[i] * 32 + bin_id + j * 32] = tables[i]->table[bin_id][j];
                }
            }
        }
        d_table_sizes = new DeviceArray<uint32_t>(dense_vertex_num + 1, context_);
        d_hash_tables = new DeviceArray<uintV>(table_sizes[dense_vertex_num] * 32, context_);
        HToD(d_table_sizes->GetArray(), table_sizes, dense_vertex_num + 1);
        HToD(d_hash_tables->GetArray(), hash_tables, table_sizes[dense_vertex_num] * 32);
    }
    HashGraph(Graph *cpu_graph, CudaContext *context)
    {
        context_ = context;
        vertex_num = cpu_graph->GetVertexCount();
        table_sizes = NULL;
        hash_tables = NULL;
        d_table_sizes = NULL;
        d_hash_tables = NULL;
        BuildHashGraph(cpu_graph);
    }
    void BuildHashGraph(Graph *cpu_graph)
    {
        uintV *cols = cpu_graph->GetCols();
        uintE *row_ptrs = cpu_graph->GetRowPtrs();
        vector<HashTable *> tables;
        table_sizes = new uint32_t[vertex_num + 1];
        table_sizes[0] = 0;
        for (uintV i = 0; i < vertex_num; i++)
        {
            HashTable *new_table = new HashTable(cols + row_ptrs[i], row_ptrs[i + 1] - row_ptrs[i]);
            tables.push_back(new_table);
            table_sizes[i + 1] = table_sizes[i] + new_table->table_size;
        }
        hash_tables = new uintV[table_sizes[vertex_num] * 32];
        printf("hash_table_total_size: %d\n", table_sizes[vertex_num]);
        memset(hash_tables, 0xFF, sizeof(uintV) * table_sizes[vertex_num] * 32);
        for (uintV i = 0; i < vertex_num; i++)
        {
            // & table = tables[i]->table;
            for (int bin_id = 0; bin_id < 32; bin_id++)
            {
                for (int j = 0; j < tables[i]->table[bin_id].size(); j++)
                {
                    hash_tables[table_sizes[i] * 32 + bin_id + j * 32] = tables[i]->table[bin_id][j];
                }
            }
        }
        d_table_sizes = new DeviceArray<uint32_t>(vertex_num + 1, context_);
        d_hash_tables = new DeviceArray<uintV>(table_sizes[vertex_num] * 32, context_);
        HToD(d_table_sizes->GetArray(), table_sizes, vertex_num + 1);
        HToD(d_hash_tables->GetArray(), hash_tables, table_sizes[vertex_num] * 32);
    }
    uintV *GetHashTables()
    {
        return d_hash_tables->GetArray();
    }
    uint32_t *GetTableSizes()
    {
        return d_table_sizes->GetArray();
    }
};

class CompactHashGraph
{
private:
    CudaContext *context_;
    uint64_t *table_sizes;
    uintV* bucket_offsets;
    uintV *buckets;
    uintV vertex_num;
    DeviceArray<uint64_t> *d_table_sizes;
    DeviceArray<uintV> *d_bucket_offsets;
    DeviceArray<uintV> *d_buckets;
public:
    uint64_t* GetTableSizes() {
        return d_table_sizes->GetArray();
    }
    uintV* GetBucketOffsets(){
        return d_bucket_offsets->GetArray();
    }
    uintV* GetBuckets(){
        return d_buckets->GetArray();
    }
    CompactHashGraph(Graph *cpu_graph, uintV *root, uintV root_num, uint32_t bin_num, CudaContext *context){
        context_ = context;
        vertex_num = cpu_graph->GetVertexCount();
        table_sizes = NULL;
        bucket_offsets = NULL;
        buckets = NULL;
        d_table_sizes = NULL;
        d_bucket_offsets = NULL;
        d_buckets = NULL;
        uintV *cols = cpu_graph->GetCols();
        uintE *row_ptrs = cpu_graph->GetRowPtrs();
        vector<HashTable *> tables;
        table_sizes = new uint64_t[root_num + 1];
        table_sizes[0] = 0;
        for (uintV i = 0; i < root_num; i++)
        {
            HashTable *new_table = new HashTable(cols + row_ptrs[root[i]], row_ptrs[root[i] + 1] - row_ptrs[root[i]], bin_num);
            tables.push_back(new_table);
            table_sizes[i + 1] = table_sizes[i] + new_table->table_size;
        }
        buckets = new uintV[table_sizes[root_num]];
        bucket_offsets = new uintV[root_num * (bin_num + 1)];
        printf("hash_table_total_size: %lu\n", table_sizes[root_num]);
        for (uintV i = 0; i < root_num; i++)
        {
            bucket_offsets[i * (bin_num + 1)] = 0;
            for (int bin_id = 0; bin_id < bin_num; bin_id++)
            {
                for (int j = 0; j < tables[i]->table[bin_id].size(); j++)
                {
                    buckets[table_sizes[i] + bucket_offsets[i * (bin_num + 1) + bin_id] + j] = tables[i]->table[bin_id][j];
                }
                bucket_offsets[i * (bin_num + 1) + bin_id + 1] = bucket_offsets[i * (bin_num + 1) + bin_id] + tables[i]->table[bin_id].size();
            }
        }
        d_table_sizes = new DeviceArray<uint64_t> (root_num + 1, context);
        d_bucket_offsets = new DeviceArray<uintV> (root_num * (bin_num + 1), context);
        d_buckets = new DeviceArray<uintV> (table_sizes[root_num], context);
        HToD(d_table_sizes->GetArray(), table_sizes, root_num + 1);
        HToD(d_bucket_offsets->GetArray(), bucket_offsets, root_num * (bin_num + 1));
        HToD(d_buckets->GetArray(), buckets, table_sizes[root_num]);
    }
};

#endif