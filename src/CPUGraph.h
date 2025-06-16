#ifndef __CPU_GRAPH_H__
#define __CPU_GRAPH_H__

#include <vector>
#include <set>
#include <queue>
#include "AbstractGraph.h"
#include "GraphIO.h"
using namespace std;
// #include "GraphPartition.h"

class Graph : public AbstractGraph {
public:
	Graph(std::string& filename, bool directed) : AbstractGraph(directed) {
		GraphIO::ReadDataFile(filename, directed, vertex_count_, edge_count_,
							row_ptrs_, cols_);
		max_degree_ = 0;
		for (int i = 0; i < vertex_count_; i++)
		{
		max_degree_ = std::max(max_degree_, (uint32_t)(row_ptrs_[i + 1] - row_ptrs_[i]));
		}
	}
  // for testing
  // Graph(std::vector<std::vector<uintV>>& data) : AbstractGraph(false) {
  //   GraphIO::ReadFromVector(data, vertex_count_, edge_count_, row_ptrs_, cols_);
  // }
  	~Graph() {}

	uintE* GetRowPtrs() const { return row_ptrs_; }
	uintV* GetCols() const { return cols_; }
	uint32_t GetMaxDegree() const{return max_degree_;}
	void SetRowPtrs(uintE* row_ptrs) { row_ptrs_ = row_ptrs; }
	void SetCols(uintV* cols) { cols_ = cols; }
  	void PeelandReorder();
	void Reorder();
	void TrianglePeel();
	void TriangleReorder();
	void TriangleOrientation();
protected:
	uintE* row_ptrs_;
	uintV* cols_;
	uint32_t max_degree_;
};

void Graph::PeelandReorder()
{
	vector<pair<size_t, uintV>> vertices;
	vector<uintV> newVid(vertex_count_);
	for (int i = 0; i < vertex_count_; i++)
	{
		vertices.push_back(make_pair(size_t(row_ptrs_[i + 1] - row_ptrs_[i]), (uintV)i));
	}
	sort(vertices.begin(), vertices.end());
	uintV current_vid = 0;
	uintE current_writing_pos = 0;
	uintV* new_cols = new uintV[edge_count_];
	uintE* new_row_ptrs = new uintE[vertex_count_];
	memset(new_cols, 0, sizeof(edge_count_) * sizeof(uintV));
	memset(new_row_ptrs, 0, sizeof(vertex_count_) * sizeof(uintE));
	for (int i = 0; i < vertex_count_; i++)
	{
		if (vertices[i].first == 0) continue;
		newVid[vertices[i].second] = current_vid++; 
	}
	current_vid = 0;
	for (int i = 0; i < vertex_count_; i++)
	{
		if (vertices[i].first == 0) continue;

		for (uintE j = 0; j < row_ptrs_[vertices[i].second + 1] - row_ptrs_[vertices[i].second]; j++)
		{
			new_cols[current_writing_pos + j] = newVid[cols_[row_ptrs_[vertices[i].second] + j]];
		}
		sort(new_cols + current_writing_pos, new_cols + current_writing_pos + row_ptrs_[vertices[i].second + 1] - row_ptrs_[vertices[i].second]);
		current_writing_pos += row_ptrs_[vertices[i].second + 1] - row_ptrs_[vertices[i].second];
		new_row_ptrs[++current_vid] = current_writing_pos;
	}
	vertex_count_ = current_vid;
	delete[] row_ptrs_;
	delete[] cols_;
	row_ptrs_ = new_row_ptrs;
	cols_ = new_cols;
	return;
}

void Graph::TrianglePeel()
{
	set<uintV> all_invalid_vertices;
	queue<uintV> invalid_vertex;
	vector<set<uintV>> currentneighbors(vertex_count_);
	for (uintV i = 0; i < vertex_count_; i++)
	{
		for (uintE j = row_ptrs_[i]; j < row_ptrs_[i + 1]; j++)
		{
			currentneighbors[i].insert(cols_[j]);
		}
		if (currentneighbors[i].size() < 2){
			invalid_vertex.push(i);
			all_invalid_vertices.insert(i);
		}
	}
	while(!invalid_vertex.empty()) {
		uintV u = invalid_vertex.front();
		invalid_vertex.pop();
		for (auto v: currentneighbors[u]) {
			currentneighbors[v].erase(u);
			if (currentneighbors[v].size() < 2 && all_invalid_vertices.find(v) == all_invalid_vertices.end()) {
				invalid_vertex.push(v);
				all_invalid_vertices.insert(v);
			}
		}
	}
	vector<uintV> old2new(vertex_count_);
	vector<uintV> new2old;
	uintE new_edge_count_ = 0;
	uintV new_id = 0;
	for (uintV i = 0; i < vertex_count_; i++)
	{
		if (all_invalid_vertices.find(i) == all_invalid_vertices.end())
		{
			old2new[i] = new_id++;
			new2old.push_back(i);
			new_edge_count_ += currentneighbors[i].size();
		}
	}
	uintE* new_row_ptrs_ = new uintE[new2old.size() + 1];
	uintV* new_cols_ = new uintV[new_edge_count_];
	uintE curr_writing_pos = 0;
	new_row_ptrs_[0] = 0;
	max_degree_ = 0;
	for (uintV i = 0; i < new2old.size(); i++)
	{
		max_degree_ = max(max_degree_, (uint32_t)(currentneighbors[new2old[i]].size()));
		for (auto u : currentneighbors[new2old[i]])
		{
			assert(all_invalid_vertices.find(u) == all_invalid_vertices.end());
			new_cols_[curr_writing_pos++] = old2new[u];
		}
		new_row_ptrs_[i + 1] = curr_writing_pos;
	}
	vertex_count_ = new2old.size();
	edge_count_ = new_edge_count_;
	delete[] row_ptrs_;
	delete[] cols_;
	row_ptrs_ = new_row_ptrs_;
	cols_ = new_cols_;
	return;
}

void Graph::TriangleReorder()
{
	vector<uintV> old2new(vertex_count_);
	vector<uintV> new2old(vertex_count_);
	vector<pair<uintV, uintV>> vertex_degree;
	for (uintV i = 0; i < vertex_count_; i++)
	{
		vertex_degree.push_back(make_pair(i, row_ptrs_[i + 1] - row_ptrs_[i]));
	}
	sort(vertex_degree.begin(), vertex_degree.end(), [](const auto& a, const auto& b) {
		return a.second == b.second ? a.first < b.first : a.second > b.second;
	});
	for (uintV i = 0; i < vertex_count_; i++)
	{
		old2new[vertex_degree[i].first] = i;
		new2old[i] = vertex_degree[i].first;
	}
	uintE* new_row_ptrs_ = new uintE[vertex_count_ + 1];
	uintV* new_cols_ = new uintV[edge_count_];
	uintE curr_writing_pos = 0;
	new_row_ptrs_[0] = 0;
	for (uintV i = 0; i < vertex_count_; i++)
	{
		for (uintE j = row_ptrs_[new2old[i]]; j < row_ptrs_[new2old[i] + 1]; j++)
		{
			new_cols_[curr_writing_pos++] = old2new[cols_[j]];
		}
		new_row_ptrs_[i + 1] = curr_writing_pos;
		sort(new_cols_ + new_row_ptrs_[i], new_cols_ + new_row_ptrs_[i + 1]);
	}
	delete[] row_ptrs_;
	delete[] cols_;
	row_ptrs_ = new_row_ptrs_;
	cols_ = new_cols_;
	return;
}

void Graph::TriangleOrientation()
{
	assert(edge_count_ % 2 == 0);
	cout << edge_count_ << endl;
	uintE* new_row_ptrs_ = new uintE[vertex_count_ + 1];
	uintV* new_cols_ = new uintV[edge_count_ / 2];
	uintE curr_writing_pos = 0;
	new_row_ptrs_[0] = 0;
	max_degree_ = 0;
	for (uintV i = 0; i < vertex_count_; i++)
	{
		for (uintE j = row_ptrs_[i]; j < row_ptrs_[i + 1]; j++)
		{
			if (cols_[j] < i)
			{
				new_cols_[curr_writing_pos++] = cols_[j];
			}
		}
		new_row_ptrs_[i + 1] = curr_writing_pos;
		max_degree_ = max(max_degree_, (uint32_t)(new_row_ptrs_[i + 1] - new_row_ptrs_[i]));
	}
	delete[] row_ptrs_;
	delete[] cols_;
	row_ptrs_ = new_row_ptrs_;
	cols_ = new_cols_;
	edge_count_ /= 2;
	return;
}

#endif
