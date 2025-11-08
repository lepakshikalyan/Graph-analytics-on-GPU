#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

_global_ void bfs_csr(int *row_ptr, int *col_idx, int *dist,
                        int *front, int fsize, int *next, int *nsize, int level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= fsize) return;
    int u = front[tid];
    for (int i = row_ptr[u]; i < row_ptr[u+1]; i++) {
        int v = col_idx[i];
        if (atomicCAS(&dist[v], -1, level+1) == -1) {
            int idx = atomicAdd(nsize,1);
            next[idx] = v;
        }
    }
}

_global_ void bfs_update_edges(int *row_ptr, int *col_idx, int *dist,
                                 int *front, int fsize, int *next, int *nsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= fsize) return;
    int u = front[tid];
    int du = dist[u];
    for (int i = row_ptr[u]; i < row_ptr[u+1]; i++) {
        int v = col_idx[i];
        int dv = dist[v];
        if (du + 1 < dv) {
            if (atomicMin(&dist[v], du+1) > du+1) {
                int idx = atomicAdd(nsize,1);
                next[idx] = v;
            }
        }
    }
}

void loadGraph(const string &file, vector<pair<int,int>> &edges, int &n) {
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
        edges.emplace_back(u,v);
        n = max(n, max(u,v));
    }
    fin.close();
    n++;
    cout << "Loaded " << n << " nodes, " << edges.size() << " edges from " << file << "\n";
}

void generateUpdates(const string &file, int n, int k) {
    ofstream fout(file);
    random_device rd; mt19937 gen(rd());
    uniform_int_distribution<> dis(0,n-1);
    for(int i=0;i<k;i++){
        int u = dis(gen), v = dis(gen);
        fout << i << " ADD " << u << " " << v << "\n";
    }
    fout.close();
    cout << "Generated " << k << " random edge additions\n";
}

void loadUpdates(const string &file, vector<pair<int,int>> &updates){
    ifstream fin(file);
    int t,u,v; string op;
    while(fin >> t >> op >> u >> v) updates.emplace_back(u,v);
    fin.close();
    cout << "Loaded " << updates.size() << " updates\n";
}

void buildCSR(int n,const vector<pair<int,int>> &edges,vector<int> &row_ptr,vector<int> &col_idx){
    row_ptr.assign(n+1,0);
    vector<int> deg(n,0);
    for(auto &[u,v]:edges) deg[u]++;
    for(int i=1;i<=n;i++) row_ptr[i]=row_ptr[i-1]+deg[i-1];
    col_idx.resize(edges.size());
    vector<int> pos(n,0);
    for(auto &[u,v]:edges) col_idx[row_ptr[u]+pos[u]++]=v;
}

int main() {

    string graphFile="data.txt";
    string updateFile="dynamic_dataset.txt";

    vector<pair<int,int>> edges;
    vector<pair<int,int>> updates;
    int n;
    loadGraph(graphFile,edges,n);

    int n_updates = 3500;
    generateUpdates(updateFile,n,n_updates);
    loadUpdates(updateFile,updates);

    vector<int> row_ptr,col_idx;
    buildCSR(n,edges,row_ptr,col_idx);

    int *d_row,*d_col,*d_dist,*d_front,*d_next,*d_next_size;
    cudaMalloc(&d_row,(n+1)*sizeof(int));
    cudaMalloc(&d_col,col_idx.size()*sizeof(int));
    cudaMalloc(&d_dist,n*sizeof(int));
    cudaMalloc(&d_front,n*sizeof(int));
    cudaMalloc(&d_next,n*sizeof(int));
    cudaMalloc(&d_next_size,sizeof(int));

    cudaMemcpy(d_row,row_ptr.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,col_idx.data(),col_idx.size()*sizeof(int),cudaMemcpyHostToDevice);

    vector<int> dist(n,-1);
    dist[0]=0;
    cudaMemcpy(d_dist,dist.data(),n*sizeof(int),cudaMemcpyHostToDevice);
    int front[1]={0};
    cudaMemcpy(d_front,front,sizeof(int),cudaMemcpyHostToDevice);

    int frontier_size=1,level=0;
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    while(frontier_size>0){
        cudaMemset(d_next_size,0,sizeof(int));
        int blocks=(frontier_size+255)/256;
        bfs_csr<<<blocks,256>>>(d_row,d_col,d_dist,d_front,frontier_size,d_next,d_next_size,level);
        cudaDeviceSynchronize();
        cudaMemcpy(&frontier_size,d_next_size,sizeof(int),cudaMemcpyDeviceToHost);
        swap(d_front,d_next);
        level++;
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float staticTime;
    cudaEventElapsedTime(&staticTime,start,end);
    cout << "Static CSR BFS rebuild time: " << staticTime << " ms\n";

    vector<pair<int,int>> edges_after = edges;
    for(auto &[u,v]: updates) edges_after.emplace_back(u,v);

    buildCSR(n,edges_after,row_ptr,col_idx);
    cudaMemcpy(d_row,row_ptr.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,col_idx.data(),col_idx.size()*sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_dist,dist.data(),n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_front,front,sizeof(int),cudaMemcpyHostToDevice);
    frontier_size=updates.size(); // all new edges as initial frontier
    vector<int> new_front;
    for(auto &[u,v]: updates) new_front.push_back(u);
    cudaMemcpy(d_front,new_front.data(),frontier_size*sizeof(int),cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    while(frontier_size>0){
        cudaMemset(d_next_size,0,sizeof(int));
        int blocks=(frontier_size+255)/256;
        bfs_update_edges<<<blocks,256>>>(d_row,d_col,d_dist,d_front,frontier_size,d_next,d_next_size);
        cudaDeviceSynchronize();
        cudaMemcpy(&frontier_size,d_next_size,sizeof(int),cudaMemcpyDeviceToHost);
        swap(d_front,d_next);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float dynamicTime;
    cudaEventElapsedTime(&dynamicTime,start,end);
    cout << "Dynamic BFS after edge additions: " << dynamicTime << " ms\n";

    cout << "Speedup: " << (staticTime/dynamicTime) << "x\n";

    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_dist);
    cudaFree(d_front);
    cudaFree(d_next);
    cudaFree(d_next_size);
    cudaDeviceReset();
    return 0;
}