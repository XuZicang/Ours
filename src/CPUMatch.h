#include "Meta.h"
#include <bits/stdc++.h>
#include "CPUGraph.h"
#include "Query.h"
using namespace std;
void backtrackingCounting(
    uintV vertex_count,
    uintE *row_ptrs,
    uintV *cols,
    uintV q_vertex_count,
    uintE *q_row_ptrs,
    uintV *q_cols,
    uintV *q_restriction,
    uintV *partial_match,
    uintV current_layer,
    uint64_t &counter)
{
    if (current_layer == q_vertex_count)
    {
        counter += 1;
        return;
    }
    uintE parent_start = q_row_ptrs[current_layer];
    uintE parent_end = q_row_ptrs[current_layer + 1];
    vector<uintV> candidates;
    uintV vid = partial_match[q_cols[parent_start]];
    for (uintE n = row_ptrs[vid]; n < row_ptrs[vid + 1]; n++)
    {
        if (q_restriction[current_layer] != 0xFFFFFFFFU && cols[n] <= partial_match[q_restriction[current_layer]])
        {
            continue;
        }
        bool flag = false;
        for (uintV i = 0; i < current_layer; i++)
        {
            if (cols[n] == partial_match[i])
            {
                flag = true;
                break;
            }
        }
        if (flag)
            continue;
        for (uintE backward_neighbor = parent_start + 1; backward_neighbor <  parent_end; backward_neighbor++)
        {
            if (q_cols[backward_neighbor] >= current_layer)
                continue;
            if (!binary_search(cols + row_ptrs[partial_match[q_cols[backward_neighbor]]],
                               cols + row_ptrs[partial_match[q_cols[backward_neighbor]] + 1], cols[n]))
            {
                flag = true;
                break;
            }
        }
        if (!flag)
        {
            candidates.push_back(cols[n]);
        }
    }
    
    for (auto v : candidates)
    {
        partial_match[current_layer] = v;
        backtrackingCounting(
            vertex_count,
            row_ptrs,
            cols,
            q_vertex_count,
            q_row_ptrs,
            q_cols,
            q_restriction,
            partial_match,
            current_layer + 1,
            counter);
    }
}

void backtrackingMatching(

)
{
}

class CPUMatch
{
    Graph *graph;
    Query *query;

public:
    CPUMatch(Graph *_graph, Query *_query) : graph(_graph), query(_query) {}
    uint64_t count();
    uint64_t match();
};

uint64_t CPUMatch::match()
{
    return 0;
}
uint64_t CPUMatch::count()
{
    uint64_t counter = 0;
    uintV *partial_match = new uintV[query->GetVertexCount()];
    for (uintV i = 0; i < graph->GetVertexCount(); i++)
    {
        partial_match[0] = i;
        backtrackingCounting(
            graph->GetVertexCount(),
            graph->GetRowPtrs(),
            graph->GetCols(),
            query->GetVertexCount(),
            query->GetRowPtrs(),
            query->GetCols(),
            query->GetRestriction(),
            partial_match,
            1,
            counter);
    }
    delete[] partial_match;
    return counter;
}