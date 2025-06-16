#ifndef __CUCKOO_CUH__
#define __CUCKOO_CUH__

#include "AbstractGraph.h"
#include "CPUGraph.h"
#include "utils/context/CudaContext.cuh"
#include "utils/DeviceArray.cuh"
#include "utils/GPUUtil.cuh"
#include "Meta.h"
#include "Hash.cuh"

class CuckooHash
{
public: 
    CuckooHash(uintV* list, uintV list_size, uintV list_v, uintV list_id)
    {
        table_size = 1;
        while (table_size < list_size * 1.05)
        {
            table_size *= 2;
        }
        table0.clear();
        table1.clear();
        table0.resize(table_size, 0xFFFFFFFFU);
        table1.resize(table_size, 0xFFFFFFFFU);
        for (uint32_t i = 0; i < list_size; i++)
        {
            if (list[i] != 0xFFFFFFFFU)
            {
                insert(list[i]);
            }
        }
        // for (uint32_t i = 0; i < list_size; i++)
        // {
        //     if (list_id == 1024)
        //     if (!(table0[hash0(list[i]) % table_size] == list[i] || table1[hash1(list[i]) % table_size] == list[i]))
        //     {
        //         cout << list[i] << " " << hash0(list[i]) << " " << hash1(list[i]) << " " << table0[hash0(list[i]) % table_size] << " " << table1[hash1(list[i]) % table_size] << endl;
        //     }
        //     else if (list[i] == 215){
        //         cout << list[i] << " " << hash0(list[i]) << " " << hash1(list[i]) << " " << table0[hash0(list[i]) % table_size] << " " << table1[hash1(list[i]) % table_size] << endl;
        //     }
        // }
        // if (table_size > 2048)
        // {
        //     cout << list_size << endl;
        // }
    }

    bool insert(uintV vid)
    {
        uintV current_vid = vid;
        int table_idx = 0;
        for (int kicks = 0; kicks < MAX_KICKS; ++kicks) {
            uint32_t h0 = hash0(current_vid) % table_size;
            uint32_t h1 = hash1(current_vid) % table_size;
            if (table0[h0] == 0xFFFFFFFFU) {
                table0[h0] = current_vid;
                return true;
            }
            if (table1[h1] == 0xFFFFFFFFU) {
                table1[h1] = current_vid;
                return true;
            }
            if (kicks % 2 == 0) {
                swap(table0[h0], current_vid);
            } else {
                swap(table1[h1], current_vid);
            }
        }
        enlarge();
    
        return insert(current_vid);
    }
    void enlarge()
    {
        vector<uintV> old_elements;
        for (auto elem : table0) {
            if (elem != 0xFFFFFFFFU) old_elements.push_back(elem);
        }
        for (auto elem : table1) {
            if (elem != 0xFFFFFFFFU) old_elements.push_back(elem);
        }

        // 扩容两倍
        table_size *= 2;
        table0.clear();
        table0.resize(table_size, 0xFFFFFFFFU);
        table1.clear();
        table1.resize(table_size, 0xFFFFFFFFU);

        // 重新插入所有元素
        for (auto elem : old_elements) {
            insert(elem);
        }
    }
    const static int MAX_KICKS = 500;
    size_t table_size;
    vector<uintV> table0;
    vector<uintV> table1;
};

class CuckooGraph
{
public:
    CuckooGraph(Graph *cpu_graph, CudaContext *context, uintV* dense_vertices, uintV dense_vertex_num) : context_(context)
    {
        hash_table_sizes = NULL;
        hash_tables = NULL;
        d_hash_table_sizes = NULL;
        d_hash_tables = NULL;
        BuildCuckooGraph(cpu_graph, context_, dense_vertices, dense_vertex_num);
    }
    ~CuckooGraph()
    {
        if (!hash_table_sizes)
        delete[] hash_table_sizes;
        if (!hash_tables)
        delete[] hash_tables;
    }
    uint32_t* GetHashTableSizes()
    {
        return d_hash_table_sizes->GetArray();
    }
    uintV* GetHashTables()
    {
        return d_hash_tables->GetArray();
    }
protected:
    CudaContext *context_;
    DeviceArray<uint32_t>*d_hash_table_sizes;
    DeviceArray<uintV> *d_hash_tables;
    uint32_t* hash_table_sizes;
    uintV* hash_tables;
    void BuildCuckooGraph(Graph *cpu_graph, CudaContext *context, uintV* dense_vertices, uintV dense_vertex_num)
    {
        cudaStream_t stream = context->Stream();
        uintV* cols = cpu_graph->GetCols();
        uintE* row_ptrs = cpu_graph->GetRowPtrs();
        std::vector<uint32_t> prefix_sums(1, 0); // 前缀和数组初始包含0
        std::vector<uintV> all_tables;
        uint32_t max_table_size = 0;
        for (uintV i = 0; i < dense_vertex_num; i++)
        {
            uintV vertex = dense_vertices[i];
            uintE start = row_ptrs[vertex];
            uintE end = row_ptrs[vertex + 1];
            uintV degree = end - start;
            
            // 创建Cuckoo哈希表
            CuckooHash hash_table(cols + start, degree, vertex, i);
            uint32_t table_size = hash_table.table_size;
            max_table_size = max(table_size, max_table_size);
            all_tables.insert(all_tables.end(), hash_table.table0.begin(), hash_table.table0.end());
            all_tables.insert(all_tables.end(), hash_table.table1.begin(), hash_table.table1.end());
            prefix_sums.push_back(prefix_sums.back() + table_size * 2);
        }
        cout << all_tables.size() << " " << prefix_sums.back();
        cout << "construct ok" << endl;
        cout << "max hash_table: " << max_table_size << endl;
        hash_table_sizes = new uint32_t[prefix_sums.size()];
        std::copy(prefix_sums.begin(), prefix_sums.end(), hash_table_sizes);
        hash_tables = new uintV[all_tables.size()];
        std::copy(all_tables.begin(), all_tables.end(), hash_tables);
        d_hash_table_sizes = new DeviceArray<uint32_t>(prefix_sums.size(), context);
        d_hash_tables = new DeviceArray<uintV>(all_tables.size(), context);
        HToD<uint32_t>(d_hash_table_sizes->GetArray(), hash_table_sizes, prefix_sums.size());
        HToD<uintV>(d_hash_tables->GetArray(), hash_tables, all_tables.size());
    }
};

#endif