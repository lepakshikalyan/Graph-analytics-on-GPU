#include<bits/stdc++.h>
#include<cuda_runtime.h>
using namespace std;

__global__ void bfs_step(const int*row_ptr,const int*col_idx,int*dist,
                         const int*front,int fsize,int*next,int*nsize){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=fsize)return;
    int u=front[tid];
    int du=dist[u];
    for(int ei=row_ptr[u];ei<row_ptr[u+1];ei++){
        int v=col_idx[ei];
        if(atomicCAS(&dist[v],-1,du+1)==-1){
            int idx=atomicAdd(nsize,1);
            next[idx]=v;
        }
    }
}

void buildCSR(int n,const vector<pair<int,int>>&edges,
              vector<int>&row_ptr,vector<int>&col_idx){
    row_ptr.assign(n+1,0);
    vector<int>deg(n,0);
    for(auto&e:edges)deg[e.first]++;
    for(int i=1;i<=n;i++)row_ptr[i]=row_ptr[i-1]+deg[i-1];
    col_idx.assign(edges.size(),0);
    vector<int>pos(n,0);
    for(auto&e:edges){
        int u=e.first,v=e.second;
        col_idx[row_ptr[u]+pos[u]++]=v;
    }
}

void loadGraph(const string&file,vector<pair<int,int>>&edges,int&n){
    ifstream fin(file);
    n=0;
    string line;
    while(getline(fin,line)){
        if(line.empty()||line[0]=='#')continue;
        stringstream ss(line);
        int u,v;
        ss>>u>>v;
        edges.emplace_back(u,v);
        if(u>n)n=u;
        if(v>n)n=v;
    }
    n++;
}

struct Update{
    int u,v;
};

void loadUpdates(const string&file,vector<Update>&updates){
    ifstream fin(file);
    int t,u,v;
    string op;
    while(fin>>t>>op>>u>>v){
        updates.push_back({u,v});
    }
}

int main(){
    string graphFile="graph.txt";
    string updateFile="updates.txt";

    vector<pair<int,int>>edges;
    int n;
    loadGraph(graphFile,edges,n);

    vector<Update>updates;
    loadUpdates(updateFile,updates);

    vector<int>row_ptr,col_idx;
    buildCSR(n,edges,row_ptr,col_idx);

    int*d_row,*d_col,*d_dist,*d_front,*d_next,*d_next_size;
    cudaMalloc(&d_row,(n+1)*sizeof(int));
    cudaMalloc(&d_col,col_idx.size()*sizeof(int));
    cudaMalloc(&d_dist,n*sizeof(int));
    cudaMalloc(&d_front,n*sizeof(int));
    cudaMalloc(&d_next,n*sizeof(int));
    cudaMalloc(&d_next_size,sizeof(int));

    cudaMemcpy(d_row,row_ptr.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,col_idx.data(),col_idx.size()*sizeof(int),cudaMemcpyHostToDevice);

    vector<int>dist(n,-1);
    dist[0]=0;
    cudaMemcpy(d_dist,dist.data(),n*sizeof(int),cudaMemcpyHostToDevice);

    vector<int>frontier={0};
    int frontier_size=1;
    cudaMemcpy(d_front,frontier.data(),sizeof(int),cudaMemcpyHostToDevice);

    int MAX_ITERS=n;

    for(int iter=0;iter<MAX_ITERS&&frontier_size>0;iter++){
        cudaMemset(d_next_size,0,sizeof(int));
        int blocks=(frontier_size+255)/256;
        bfs_step<<<blocks,256>>>(d_row,d_col,d_dist,d_front,
                                 frontier_size,d_next,d_next_size);
        cudaDeviceSynchronize();
        cudaMemcpy(&frontier_size,d_next_size,sizeof(int),cudaMemcpyDeviceToHost);
        swap(d_front,d_next);
    }

    cudaMemcpy(dist.data(),d_dist,n*sizeof(int),cudaMemcpyDeviceToHost);

    unordered_map<long long,int>emap;
    emap.reserve(edges.size()*2);
    for(auto&e:edges){
        long long k=((long long)e.first<<32)|e.second;
        emap[k]=1;
    }
    for(auto&u:updates){
        long long k=((long long)u.u<<32)|u.v;
        emap[k]=1;
    }

    vector<pair<int,int>>edges_after;
    edges_after.reserve(emap.size());
    for(auto&p:emap){
        int u=p.first>>32;
        int v=p.first&0xffffffff;
        edges_after.emplace_back(u,v);
    }

    cudaEvent_t s1,e1,s2,e2;
    cudaEventCreate(&s1);
    cudaEventCreate(&e1);
    cudaEventCreate(&s2);
    cudaEventCreate(&e2);

    cudaEventRecord(s1);

    vector<char>inF(n,0);
    vector<int>dyn_front;

    for(auto&u:updates){
        int a=u.u,b=u.v;
        if(dist[a]!=-1&&(dist[b]==-1||dist[a]+1<dist[b])){
            dist[b]=dist[a]+1;
            if(!inF[b]){
                inF[b]=1;
                dyn_front.push_back(b);
            }
        }
    }

    frontier_size=dyn_front.size();
    cudaMemcpy(d_dist,dist.data(),n*sizeof(int),cudaMemcpyHostToDevice);

    if(frontier_size>0){
        cudaMemcpy(d_front,dyn_front.data(),frontier_size*sizeof(int),cudaMemcpyHostToDevice);
    }

    for(int iter=0;iter<MAX_ITERS&&frontier_size>0;iter++){
        cudaMemset(d_next_size,0,sizeof(int));
        int blocks=(frontier_size+255)/256;
        bfs_step<<<blocks,256>>>(d_row,d_col,d_dist,d_front,
                                 frontier_size,d_next,d_next_size);
        cudaDeviceSynchronize();
        cudaMemcpy(&frontier_size,d_next_size,sizeof(int),cudaMemcpyDeviceToHost);
        swap(d_front,d_next);
    }

    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float T1;
    cudaEventElapsedTime(&T1,s1,e1);

    cudaEventRecord(s2);

    buildCSR(n,edges_after,row_ptr,col_idx);
    cudaMemcpy(d_row,row_ptr.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,col_idx.data(),col_idx.size()*sizeof(int),cudaMemcpyHostToDevice);

    for(int i=0;i<n;i++)dist[i]=-1;
    dist[0]=0;
    cudaMemcpy(d_dist,dist.data(),n*sizeof(int),cudaMemcpyHostToDevice);

    frontier={0};
    frontier_size=1;
    cudaMemcpy(d_front,frontier.data(),sizeof(int),cudaMemcpyHostToDevice);

    for(int iter=0;iter<MAX_ITERS&&frontier_size>0;iter++){
        cudaMemset(d_next_size,0,sizeof(int));
        int blocks=(frontier_size+255)/256;
        bfs_step<<<blocks,256>>>(d_row,d_col,d_dist,d_front,
                                 frontier_size,d_next,d_next_size);
        cudaDeviceSynchronize();
        cudaMemcpy(&frontier_size,d_next_size,sizeof(int),cudaMemcpyDeviceToHost);
        swap(d_front,d_next);
    }

    cudaEventRecord(e2);
    cudaEventSynchronize(e2);

    float T2;
    cudaEventElapsedTime(&T2,s2,e2);

    cout<<"Dynamic BFS Time="<<T1<<" ms\n";
    cout<<"Rebuild+Full BFS Time="<<T2<<" ms\n";
    cout<<"Speedup="<<T2/T1<<"x\n";

    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_dist);
    cudaFree(d_front);
    cudaFree(d_next);
    cudaFree(d_next_size);

    return 0;
}
