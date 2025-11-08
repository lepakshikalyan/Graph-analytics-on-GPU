#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

__global__ void pagerank_kernel(int n, const int *row_ptr, const int *col_idx,
                                const float *old_rank, float *new_rank,
                                const float *inv_outdeg, float damping) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;

    float sum = 0.0f;
    for (int i = row_ptr[u]; i < row_ptr[u + 1]; i++) {
        int v = col_idx[i];
        sum += old_rank[v] * inv_outdeg[v];
    }
    new_rank[u] = (1.0f - damping) / n + damping * sum;
}

__global__ void dynamic_pagerank_update(int n, const int *row_ptr, const int *col_idx,
                                        float *rank, const float *inv_outdeg,
                                        const int *affected_nodes, int affected_size,
                                        float damping) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= affected_size) return;

    int u = affected_nodes[tid];
    float sum = 0.0f;
    for (int i = row_ptr[u]; i < row_ptr[u + 1]; i++) {
        int v = col_idx[i];
        sum += rank[v] * inv_outdeg[v];
    }
    rank[u] = (1.0f - damping) / n + damping * sum;
}

void loadGraph(const string &file, vector<pair<int, int>> &edges, int &n) {
    ifstream fin(file);
    if (!fin.is_open()) {
        cerr << "Failed to open " << file << "\n";
        exit(1);
    }
    n = 0;
    string line;
    while (getline(fin, line)) {
        if (line.empty()||line[0]=='#') continue;
        stringstream ss(line);
        int u, v;
        ss >> u >> v;
        edges.emplace_back(u, v);
        n = max(n, max(u, v));
    }
    fin.close();
    n++;
    cout << "Loaded " << n << " nodes, " << edges.size() << " edges from " << file << "\n";
}

void buildCSR(int n, const vector<pair<int, int>> &edges, vector<int> &row_ptr, vector<int> &col_idx) {
    row_ptr.assign(n + 1, 0);
    vector<int> indeg(n, 0);
    for (auto &[u, v] : edges) indeg[v]++;
    for (int i = 1; i <= n; i++) row_ptr[i] = row_ptr[i - 1] + indeg[i - 1];
    col_idx.resize(edges.size());
    vector<int> pos(n, 0);
    for (auto &[u, v] : edges) col_idx[row_ptr[v] + pos[v]++] = u;
}

void generateSequentialUpdates(const string &file, int n, int k) {
    ofstream fout(file);
    int u = 0, v = 1;
    for (int i = 0; i < k; i++) {
        fout << i << " ADD " << u << " " << v << "\n";
        u = (u + 1) % n;
        v = (v + 2) % n;
    }
    fout.close();
    cout << "Generated " << k << " sequential edge additions\n";
}

void loadUpdates(const string &file, vector<pair<int, int>> &updates) {
    ifstream fin(file);
    int t, u, v;
    string op;
    while (fin >> t >> op >> u >> v) updates.emplace_back(u, v);
    fin.close();
    cout << "Loaded " << updates.size() << " updates\n";
}

int main() {
    string graphFile = "data.txt";
    string updateFile = "dynamic_dataset.txt";

    vector<pair<int, int>> edges, updates;
    int n;
    loadGraph(graphFile, edges, n);

    int n_updates = 2000;
    generateSequentialUpdates(updateFile, n, n_updates);
    loadUpdates(updateFile, updates);

    
    vector<int> row_ptr, col_idx;
    buildCSR(n, edges, row_ptr, col_idx);

    vector<int> outdeg(n, 0);
    for (auto &[u, v] : edges) outdeg[u]++;
    vector<float> inv_outdeg(n);
    for (int i = 0; i < n; i++)
        inv_outdeg[i] = (outdeg[i] > 0) ? 1.0f / outdeg[i] : 0.0f;

   
    int *d_row, *d_col;
    float *d_rank_old, *d_rank_new, *d_inv_outdeg;
    cudaMalloc(&d_row, (n + 1) * sizeof(int));
    cudaMalloc(&d_col, col_idx.size() * sizeof(int));
    cudaMalloc(&d_rank_old, n * sizeof(float));
    cudaMalloc(&d_rank_new, n * sizeof(float));
    cudaMalloc(&d_inv_outdeg, n * sizeof(float));

    cudaMemcpy(d_row, row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col_idx.data(), col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_outdeg, inv_outdeg.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    vector<float> rank(n, 1.0f / n);
    cudaMemcpy(d_rank_old, rank.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    float damping = 0.85f;
    int max_iter = 30;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int iter = 0; iter < max_iter; iter++) {
        pagerank_kernel<<<blocks, threads>>>(n, d_row, d_col, d_rank_old, d_rank_new, d_inv_outdeg, damping);
        cudaDeviceSynchronize();
        swap(d_rank_old, d_rank_new);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float staticTime;
    cudaEventElapsedTime(&staticTime, start, end);
    cout << "Starting PageRank time: " << staticTime << " ms\n";
    vector<pair<int, int>> edges_after = edges;
    for (auto &[u, v] : updates) edges_after.emplace_back(u, v);
    buildCSR(n, edges_after, row_ptr, col_idx);
    cudaMemcpy(d_row, row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col_idx.data(), col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);

    vector<int> affected;
    for (auto &[u, v] : updates) affected.push_back(v);
    sort(affected.begin(), affected.end());
    affected.erase(unique(affected.begin(), affected.end()), affected.end());
    int affected_size = affected.size();

    int *d_affected;
    cudaMalloc(&d_affected, affected_size * sizeof(int));
    cudaMemcpy(d_affected, affected.data(), affected_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    int dyn_blocks = (affected_size + threads - 1) / threads;
    for (int iter = 0; iter < 10; iter++) { 
        dynamic_pagerank_update<<<dyn_blocks, threads>>>(n, d_row, d_col, d_rank_old, d_inv_outdeg,
                                                         d_affected, affected_size, damping);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float dynamicTime;
    cudaEventElapsedTime(&dynamicTime, start, end);
    cout << "Dynamic PageRank update time: " << dynamicTime << " ms\n";


    cudaMemcpy(rank.data(), d_rank_old, n * sizeof(float), cudaMemcpyDeviceToHost);

   
    vector<pair<float, int>> nodes;
    for (int i = 0; i < n; i++) nodes.emplace_back(rank[i], i);
    sort(nodes.rbegin(), nodes.rend());
    cout << "Top 10 PageRank nodes:\n";
    for (int i = 0; i < min(10, n); i++)
        cout << "Node " << nodes[i].second << ": " << nodes[i].first << "\n";

    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_rank_old);
    cudaFree(d_rank_new);
    cudaFree(d_inv_outdeg);
    cudaFree(d_affected);
    cudaDeviceReset();

    return 0;
}