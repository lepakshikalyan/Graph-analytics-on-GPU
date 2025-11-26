#include<bits/stdc++.h>
#include<cuda_runtime.h>
using namespace std;

__global__ void gpu_wcc_kernel(const int *row_ptr,const int *col_idx,int *comp,int n,const int *front,int fsize,int *next,int *nsize){
int tid=blockIdx.x*blockDim.x+threadIdx.x;
if(tid>=fsize)return;
int u=front[tid];
int cu=comp[u];
for(int ei=row_ptr[u];ei<row_ptr[u+1];++ei){
int v=col_idx[ei];
if(v>=n)continue;
int old=comp[v];
if(old>cu){
int prev=atomicMin(&comp[v],cu);
if(prev>cu){
int idx=atomicAdd(nsize,1);
next[idx]=v;
}
}
}
}

void buildCSR(int n,const vector<tuple<int,int,int>> &edges,vector<int> &row_ptr,vector<int> &col_idx){
row_ptr.assign(n+1,0);
vector<int> deg(n,0);
for(auto &e:edges)deg[get<0>(e)]++;
for(int i=1;i<=n;i++)row_ptr[i]=row_ptr[i-1]+deg[i-1];
col_idx.assign(edges.size(),0);
vector<int> pos(n,0);
for(auto &e:edges){
int u=get<0>(e);
int v=get<1>(e);
int id=row_ptr[u]+pos[u]++;
col_idx[id]=v;
}
}

void loadGraph(const string &file,vector<tuple<int,int,int>> &edges,int &n){
ifstream fin(file);
n=0;
string line;
while(getline(fin,line)){
if(line.empty()||line[0]=='#')continue;
stringstream ss(line);
int u,v;
ss>>u>>v;
edges.emplace_back(u,v,0);
n=max(n,max(u,v));
}
n++;
}

int main(){
string graphFile="wcc.txt";
string updateFile="updates.txt";
vector<tuple<int,int,int>> edges;
int n;
loadGraph(graphFile,edges,n);

vector<tuple<int,int,int>> updates;
ifstream fin(updateFile);
while(true){
int u,v;
if(!(fin>>u>>v))break;
updates.emplace_back(u,v,0);
}

unordered_map<long long,int> edge_map;
edge_map.reserve(edges.size()*2);
for(auto &e:edges)edge_map[((long long)get<0>(e)<<32)|get<1>(e)]=0;
for(auto &u:updates)edge_map[((long long)get<0>(u)<<32)|get<1>(u)]=0;

vector<tuple<int,int,int>> edges_after;
edges_after.reserve(edge_map.size());
for(auto &p:edge_map){
int u=p.first>>32;
int v=p.first&0xffffffff;
edges_after.emplace_back(u,v,0);
}

vector<int> row_ptr,col_idx;
buildCSR(n,edges_after,row_ptr,col_idx);

vector<int> comp(n);
for(int i=0;i<n;i++)comp[i]=i;

int *d_row,*d_col,*d_comp,*d_front,*d_next,*d_next_size;
cudaMalloc(&d_row,(n+1)*sizeof(int));
cudaMalloc(&d_col,col_idx.size()*sizeof(int));
cudaMalloc(&d_comp,n*sizeof(int));
cudaMalloc(&d_front,n*sizeof(int));
cudaMalloc(&d_next,col_idx.size()*sizeof(int));
cudaMalloc(&d_next_size,sizeof(int));

cudaMemcpy(d_row,row_ptr.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice);
cudaMemcpy(d_col,col_idx.data(),col_idx.size()*sizeof(int),cudaMemcpyHostToDevice);
cudaMemcpy(d_comp,comp.data(),n*sizeof(int),cudaMemcpyHostToDevice);

cudaEvent_t s1,e1,s2,e2;
cudaEventCreate(&s1);
cudaEventCreate(&e1);
cudaEventCreate(&s2);
cudaEventCreate(&e2);

// STATIC GPU WCC
vector<int> frontier(n);
iota(frontier.begin(),frontier.end(),0);
int frontier_size=n;
cudaMemcpy(d_front,frontier.data(),n*sizeof(int),cudaMemcpyHostToDevice);
cudaEventRecord(s1);
int MAX_ITERS=n;
for(int iter=0;iter<MAX_ITERS&&frontier_size>0;iter++){
cudaMemset(d_next_size,0,sizeof(int));
int blocks=(frontier_size+255)/256;
gpu_wcc_kernel<<<blocks,256>>>(d_row,d_col,d_comp,n,d_front,frontier_size,d_next,d_next_size);
cudaDeviceSynchronize();
cudaMemcpy(&frontier_size,d_next_size,sizeof(int),cudaMemcpyDeviceToHost);
swap(d_front,d_next);
}
cudaEventRecord(e1);
cudaEventSynchronize(e1);
float static_time;
cudaEventElapsedTime(&static_time,s1,e1);
cudaMemcpy(comp.data(),d_comp,n*sizeof(int),cudaMemcpyDeviceToHost);

// CPU WCC
vector<int> comp_cpu(n);
for(int i=0;i<n;i++)comp_cpu[i]=i;
auto cpu_start=chrono::high_resolution_clock::now();
bool changed=true;
while(changed){
changed=false;
for(auto &e:edges_after){
int u=get<0>(e);
int v=get<1>(e);
if(comp_cpu[v]>comp_cpu[u]){comp_cpu[v]=comp_cpu[u];changed=true;}
if(comp_cpu[u]>comp_cpu[v]){comp_cpu[u]=comp_cpu[v];changed=true;}
}
}
auto cpu_end=chrono::high_resolution_clock::now();
double cpu_time=chrono::duration<double,chrono::milliseconds::period>(cpu_end-cpu_start).count();

// DYNAMIC GPU WCC (simulate updates)
cudaMemcpy(d_comp,comp.data(),n*sizeof(int),cudaMemcpyHostToDevice);
vector<int> dyn_frontier;
for(auto &u:updates){
int b=get<1>(u);
dyn_frontier.push_back(b);
}
frontier_size=dyn_frontier.size();
if(frontier_size)cudaMemcpy(d_front,dyn_frontier.data(),frontier_size*sizeof(int),cudaMemcpyHostToDevice);
cudaEventRecord(s2);
for(int iter=0;iter<MAX_ITERS&&frontier_size>0;iter++){
cudaMemset(d_next_size,0,sizeof(int));
int blocks=(frontier_size+255)/256;
gpu_wcc_kernel<<<blocks,256>>>(d_row,d_col,d_comp,n,d_front,frontier_size,d_next,d_next_size);
cudaDeviceSynchronize();
cudaMemcpy(&frontier_size,d_next_size,sizeof(int),cudaMemcpyDeviceToHost);
swap(d_front,d_next);
}
cudaEventRecord(e2);
cudaEventSynchronize(e2);
float dynamic_time;
cudaEventElapsedTime(&dynamic_time,s2,e2);

cout<<"CPU WCC Time="<<cpu_time<<" ms\n";
cout<<"Static GPU WCC Time="<<static_time<<" ms\n";
cout<<"Dynamic GPU WCC Time="<<dynamic_time<<" ms\n";
cout<<"Speedup(CPU/Dynamic GPU)="<<cpu_time/dynamic_time<<"x\n";

cudaFree(d_row);
cudaFree(d_col);
cudaFree(d_comp);
cudaFree(d_front);
cudaFree(d_next);
cudaFree(d_next_size);
cudaDeviceReset();
return 0;
}

