#ifndef __QUERY_H__
#define __QUERY_H__
#include "Meta.h"
#include "GraphIO.h"
#include <string>
#include <queue>
#include "CPUGraph.h"
#include <tuple>
using namespace std;
bool MatchLess(vector<uintV>& m1, vector<uintV> & m2)
{
    int p = 0;
    while (m1[p] == m2[p])
    {
        p++;
    }
    return m1[p] < m2[p];
}

bool is_adj(const uintV* adj_list, uintE list_size, uintV vid)
{
    int l = 0, r = list_size - 1;
    while (l <= r) {
        uintE mid = (l + r) / 2;
        if (adj_list[mid] == vid)
        {
            return true;
        }
        if (adj_list[mid] < vid)
            l = mid + 1;
        if (adj_list[mid] > vid)
            r = mid - 1;
    }
    return false;
}

class Query : public AbstractGraph
{

public:
    Query(std::string filename, bool directed = false) : AbstractGraph(directed)
    {
        GraphIO::ReadDataFile(filename, directed, vertex_count_, edge_count_, row_ptrs_, cols_);
        reorder();
        BuildRestriction();
    }
    uintE *GetRowPtrs() const { return row_ptrs_; };
    uintV *GetCols() const { return cols_; }
    uintV *GetRestriction() const { return restriction_; }
    bool ExistCentral() {return exist_central_node;}

protected:
    uintE *row_ptrs_;
    uintV *cols_;
    uintV *restriction_;
    bool exist_central_node;
    void reorder()
    {
        // vector<bool> visited(vertex_count_, false);
        vector<uintV> order(vertex_count_);
        vector<uintV> reverse_order(vertex_count_);
        size_t max_degree = 0;
        uintV root = 0;
        for (uintV i = 0; i < vertex_count_; i++) {
            size_t degree = row_ptrs_[i + 1] - row_ptrs_[i];
            if (max_degree < degree)
            {
                max_degree = degree;
                root = i;
            }
        }
        if (max_degree == vertex_count_ - 1) exist_central_node = true;
        else exist_central_node = false;
        queue<uintV> queue;
        vector<bool> visited(vertex_count_, false);
        queue.push(root);
        visited[root] = true;
        uintV new_vid = 0;
        while (!queue.empty()) {
            size_t size = queue.size();
            vector<uintV> same_level_vertices;
            for (size_t i = 0; i < size; i++)
            {
                uintV front = queue.front();
                same_level_vertices.push_back(front);
                queue.pop();
                for (uintV j = 0; j < vertex_count_; j++)
                {
                    if (is_adj(cols_ + row_ptrs_[front], row_ptrs_[front + 1] - row_ptrs_[front], j) && !visited[j])
                    {
                        visited[j] = true;
                        queue.push(j);
                    }
                }
            }
            vector<tuple<size_t, size_t, int>> weights;
            for (size_t i = 0; i < size; i++) {
                uintV v = same_level_vertices[i];
                size_t connections = 0;
                size_t all_connections = 0;
                for (uintV j = 0; j < vertex_count_; j++)
                {
                    if (is_adj(cols_ + row_ptrs_[v], row_ptrs_[v + 1] - row_ptrs_[v], j)) {
                        all_connections++;
                        if (visited[j]) connections++;
                    }
                }
                weights.emplace_back(all_connections, connections, v);
            }
            std::sort(weights.begin(), weights.end(), [](const auto& a, const auto& b) {
                if (std::get<0>(a) != std::get<0>(b))
                  return std::get<0>(a) > std::get<0>(b);
                else if (std::get<1>(a) != std::get<1>(b))
                  return std::get<1>(a) > std::get<1>(b);
                else if (std::get<2>(a) != std::get<2>(b))
                  return std::get<2>(a) < std::get<2>(b);
                return false;
            });
            for (const auto& w: weights) {
                order[new_vid] = get<2>(w);
                reverse_order[get<2>(w)] = new_vid;
                new_vid++;
            }
        }
        uintE* new_row_ptrs_ = new uintE[vertex_count_ + 1];
        uintV* new_cols_ = new uintV[edge_count_];
        uintE curr_pos = 0;
        new_row_ptrs_[0] = curr_pos;
        for (uintV i = 0; i < vertex_count_; i++)
        {
            for (uintE j = row_ptrs_[order[i]]; j < row_ptrs_[order[i] + 1]; j++)
            {
                new_cols_[curr_pos++] = reverse_order[cols_[j]];
            }
            new_row_ptrs_[i + 1] = curr_pos; 
        }
        memcpy(cols_, new_cols_, edge_count_ * sizeof(uintV));
        memcpy(row_ptrs_, new_row_ptrs_, (vertex_count_ + 1) * sizeof(uintE));
        delete[] new_row_ptrs_;
        delete[] new_cols_;
        return;
    }
    void BuildRestriction()
    {
        restriction_ = new uintV[vertex_count_];
        for (int i = 0; i < vertex_count_; i++)
            restriction_[i] = 0xFFFFFFFFU;
        vector<vector<uintV>> degreeEq;
        vector<vector<uintV>> cond;
        cond.resize(vertex_count_);
        vector<vector<uintV>> aut;
        FindAut(degreeEq, aut);
        for (uintV v = 0; v < vertex_count_; v++)
        {
            for (auto match: aut)
            {
                if (match[v] != v)
                {
                    cond[v].push_back(match[v]);
                }
            }
            for (auto match = aut.begin(); match != aut.end();)
            {
                if ((*match)[v] != v)
                    match = aut.erase(match);
                else
                    match++;
            }
        }
        for (uintV v = 0; v < vertex_count_; v++)
        {
            for (uintV p = v - 1;; p--)
            {
                if (p == 0xFFFFFFFFU)
                    break;
                if (find(cond[p].begin(), cond[p].end(), v) != cond[p].end())
                {
                    restriction_[v] = p;
                    break;
                }
            }
        }
        for (uintV i = 0; i < vertex_count_; i++)
            cout << "restriction[" << i << "]: " << restriction_[i] << endl;
    }
    void FindAut(vector<vector<uintV>>& degreeEq, vector<vector<uintV>>& aut)
    {
        for (uintV i = 0; i < vertex_count_; i++)
        {
            if (row_ptrs_[i + 1] - row_ptrs_[i] + 1 > degreeEq.size())
                degreeEq.resize(row_ptrs_[i + 1] - row_ptrs_[i] + 1);
            degreeEq[row_ptrs_[i + 1] - row_ptrs_[i]].emplace_back(i);
        }
        // vector<vector<uintV>> aut;
        uintV *mapping = new uintV[vertex_count_];
        DFSFindAut(degreeEq, aut, 0, mapping);
        sort(aut.begin(), aut.end(), MatchLess);
    }

    void DFSFindAut(vector<vector<uintV>> &degreeEq, vector<vector<uintV>> &aut, uintV current_vertex, uintV *mapping)
    {
        if (current_vertex == vertex_count_)
        {
            aut.push_back(vector<uintV>());
            for (uintV i = 0; i < vertex_count_; i++)
            {
                aut[aut.size() - 1].push_back(mapping[i]);
            }
            return;
        }
        uintE degree = row_ptrs_[current_vertex + 1] - row_ptrs_[current_vertex];
        for (auto v : degreeEq[degree])
        {
            // cout << "current_vertex: " << current_vertex << " " << v << endl;
            bool possible = true;
            for (uintV p = 0; p < current_vertex; p++)
            {
                if (v == mapping[p])
                {
                    possible = false;
                    break;
                }
                bool exist1 = false, exist2 = false;
                for (uintE k = 0; k < degree; k++)
                {
                    exist1 = exist1 || cols_[row_ptrs_[current_vertex] + k] == p;
                    exist2 = exist2 || cols_[row_ptrs_[v] + k] == mapping[p];
                }
                possible = possible && exist1 == exist2;
            }
            if (possible) {
                mapping[current_vertex] = v;
                DFSFindAut(degreeEq, aut, current_vertex + 1, mapping);
            }
        }
    }
};

#endif