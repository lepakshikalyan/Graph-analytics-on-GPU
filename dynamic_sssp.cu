#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

__global__ void gpu_relax(const int *row_ptr,const int *col_idx,const int *col_w,int *dist,const int *front,int fsize,int *next,int *nsize){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=fsize) return;
    int u=front[tid];
    int du=dist[u];
    if(du==INT_MAX) return;
    for(int ei=row_ptr[u];ei<row_ptr[u+1];++ei){
        int v=col_idx[ei];
        int w=col_w[ei];
        int newd=du+w;
        int old=atomicMin(&dist[v],newd);
        if(old>newd){
            int idx=atomicAdd(nsize,1);
            next[idx]=v;
        }
    }
}

void buildCSR(int n,const vector<tuple<int,int,int>> &edges,vector<int> &row_ptr,vector<int> &col_idx,vector<int> &col_w){
    row_ptr.assign(n+1,0);
    vector<int> deg(n,0);
    for(auto &e:edges) deg[get<0>(e)]++;
    for(int i=1;i<=n;i++) row_ptr[i]=row_ptr[i-1]+deg[i-1];
    col_idx.assign(edges.size(),0);
    col_w.assign(edges.size(),0);
    vector<int> pos(n,0);
    for(auto &e:edges){
        int u=get<0>(e);
        int v=get<1>(e);
        int w=get<2>(e);
        int id=row_ptr[u]+pos[u]++;
        col_idx[id]=v;
        col_w[id]=w;
    }
}

void loadGraph(const string &file,vector<tuple<int,int,int>> &edges,int &n){
    ifstream fin(file);
    n=0;
    string line;
    while(getline(fin,line)){
        if(line.empty()||line[0]=='#') continue;
        stringstream ss(line);
        int u,v,w;
        ss>>u>>v>>w;
        edges.emplace_back(u,v,w);
        n=max(n,max(u,v));
    }
    n++;
}

struct Update{string op;int u,v,w;};

void loadUpdates(const string &file,vector<Update> &updates){
    ifstream fin(file);
    while(true){
        Update u;
        if(!(fin>>u.op>>u.u>>u.v>>u.w)) break;
        updates.push_back(u);
    }
}

int main(){
    string graphFile="weighted_data.txt";
    string updateFile="updates.txt";
    vector<tuple<int,int,int>> edges;
    int n;
    loadGraph(graphFile,edges,n);
    cout<<"Nodes = "<<n<<", Edges = "<<edges.size()<<endl;
    vector<Update> updates;
    loadUpdates(updateFile,updates);
    cout<<"Loaded updates = "<<updates.size()<<endl;

    unordered_map<long long,int> edge_map;
    edge_map.reserve(edges.size()*2);
    for(auto &e:edges){
        long long key=((long long)get<0>(e)<<32)|get<1>(e);
        edge_map[key]=get<2>(e);
    }
    for(auto &u:updates){
        long long key=((long long)u.u<<32)|u.v;
        edge_map[key]=u.w;
    }

    vector<tuple<int,int,int>> edges_after;
    edges_after.reserve(edge_map.size());
    for(auto &p:edge_map){
        int u=p.first>>32;
        int v=p.first&0xffffffff;
        edges_after.emplace_back(u,v,p.second);
    }
    cout<<"Final edges after updates = "<<edges_after.size()<<endl;

    vector<int> row_ptr,col_idx,col_w;
    buildCSR(n,edges_after,row_ptr,col_idx,col_w);
    vector<int> dist(n,INT_MAX);
    dist[0]=0;

    int *d_row,*d_col,*d_w,*d_dist,*d_front,*d_next,*d_next_size;
    cudaMalloc(&d_row,(n+1)*sizeof(int));
    cudaMalloc(&d_col,col_idx.size()*sizeof(int));
    cudaMalloc(&d_w,col_w.size()*sizeof(int));
    cudaMalloc(&d_dist,n*sizeof(int));
    cudaMalloc(&d_front,n*sizeof(int));
    cudaMalloc(&d_next,n*sizeof(int));
    cudaMalloc(&d_next_size,sizeof(int));

    cudaMemcpy(d_row,row_ptr.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,col_idx.data(),col_idx.size()*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_w,col_w.data(),col_w.size()*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist,dist.data(),n*sizeof(int),cudaMemcpyHostToDevice);

    cudaEvent_t s1,e1,s2,e2;
    cudaEventCreate(&s1);
    cudaEventCreate(&e1);
    cudaEventCreate(&s2);
    cudaEventCreate(&e2);

    vector<int> frontier={0};
    int frontier_size=1;
    cudaMemcpy(d_front,frontier.data(),sizeof(int),cudaMemcpyHostToDevice);
    cudaEventRecord(s1);
    int MAX_ITERS=n;
    for(int iter=0;iter<MAX_ITERS&&frontier_size>0;iter++){
        cudaMemset(d_next_size,0,sizeof(int));
        int blocks=(frontier_size+255)/256;
        gpu_relax<<<blocks,256>>>(d_row,d_col,d_w,d_dist,d_front,frontier_size,d_next,d_next_size);
        cudaDeviceSynchronize();
        cudaMemcpy(&frontier_size,d_next_size,sizeof(int),cudaMemcpyDeviceToHost);
        swap(d_front,d_next);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float static_time;
    cudaEventElapsedTime(&static_time,s1,e1);
    cout<<"Static SSSP Time = "<<static_time<<" ms\n";
    cudaMemcpy(dist.data(),d_dist,n*sizeof(int),cudaMemcpyDeviceToHost);

    cudaEventRecord(s2);
    vector<char> inF(n,0);
    vector<int> dyn_frontier;
    for(auto &u:updates){
        int a=u.u,b=u.v,w=u.w;
        if(dist[a]!=INT_MAX&&dist[a]+w<dist[b]){
            if(!inF[b]){
                inF[b]=1;
                dyn_frontier.push_back(b);
            }
        }
    }
    frontier_size=dyn_frontier.size();
    if(frontier_size)
        cudaMemcpy(d_front,dyn_frontier.data(),frontier_size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist,dist.data(),n*sizeof(int),cudaMemcpyHostToDevice);
    for(int iter=0;iter<MAX_ITERS&&frontier_size>0;iter++){
        cudaMemset(d_next_size,0,sizeof(int));
        int blocks=(frontier_size+255)/256;
        gpu_relax<<<blocks,256>>>(d_row,d_col,d_w,d_dist,d_front,frontier_size,d_next,d_next_size);
        cudaDeviceSynchronize();
        cudaMemcpy(&frontier_size,d_next_size,sizeof(int),cudaMemcpyDeviceToHost);
        swap(d_front,d_next);
    }
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    float dynamic_time;
    cudaEventElapsedTime(&dynamic_time,s2,e2);
    cout<<"Dynamic SSSP Time = "<<dynamic_time<<" ms\n";
    cout<<"Speedup = "<<static_time/dynamic_time<<"x\n";

    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_w);
    cudaFree(d_dist);
    cudaFree(d_front);
    cudaFree(d_next);
    cudaFree(d_next_size);
    cudaDeviceReset();
    return 0;
}

