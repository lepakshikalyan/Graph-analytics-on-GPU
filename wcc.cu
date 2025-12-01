#include<bits/stdc++.h>
#include<cuda_runtime.h>
using namespace std;

__global__ void gpu_wcc_kernel(const int*row_ptr,const int*col_idx,int*comp,int n,const int*front,int fsize,int*next,int*nsize){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=fsize)return;
    int u=front[tid],cu=comp[u];
    for(int ei=row_ptr[u];ei<row_ptr[u+1];ei++){
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

void buildCSR(int n,const vector<pair<int,int>>&edges,vector<int>&row_ptr,vector<int>&col_idx){
    row_ptr.assign(n+1,0);
    for(auto&e:edges)row_ptr[e.first+1]++;
    for(int i=1;i<=n;i++)row_ptr[i]+=row_ptr[i-1];
    col_idx.assign(edges.size(),0);
    vector<int>pos(n,0);
    for(auto&e:edges){
        int u=e.first,v=e.second,id=row_ptr[u]+pos[u]++;
        col_idx[id]=v;
    }
}

void loadGraph(const string&file,vector<pair<int,int>>&edges,int&n){
    ifstream fin(file);n=0;string line;
    while(getline(fin,line)){
        if(line.empty()||line[0]=='#')continue;
        stringstream ss(line);int u,v;ss>>u>>v;
        edges.push_back({u,v});
        edges.push_back({v,u});
        n=max(n,max(u,v));
    }
    n++;
}

int main(){
    string graphFile="graph_95.txt",updateFile="updates_5.txt";

    vector<pair<int,int>>edges_orig;
    int n;
    loadGraph(graphFile,edges_orig,n);

    vector<pair<int,int>>updates;
    { ifstream fin(updateFile); int u,v; while(fin>>u>>v){ updates.push_back({u,v}); updates.push_back({v,u}); n=max(n,max(u,v)); } }

    vector<pair<int,int>>edges_after=edges_orig;
    for(auto&e:updates)edges_after.push_back(e);

    vector<int>row_ptr0,col_idx0;
    buildCSR(n,edges_orig,row_ptr0,col_idx0);

    vector<int>comp(n);
    for(int i=0;i<n;i++)comp[i]=i;

    int*d_row,*d_col,*d_comp,*d_front,*d_next,*d_next_size;
    cudaMalloc(&d_row,(n+1)*sizeof(int));
    cudaMalloc(&d_col,col_idx0.size()*sizeof(int));
    cudaMalloc(&d_comp,n*sizeof(int));
    cudaMalloc(&d_front,n*sizeof(int));
    cudaMalloc(&d_next,col_idx0.size()*sizeof(int));
    cudaMalloc(&d_next_size,sizeof(int));

    cudaMemcpy(d_row,row_ptr0.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,col_idx0.data(),col_idx0.size()*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_comp,comp.data(),n*sizeof(int),cudaMemcpyHostToDevice);

    vector<int>front(n);
    iota(front.begin(),front.end(),0);
    int fsize=n;
    cudaMemcpy(d_front,front.data(),n*sizeof(int),cudaMemcpyHostToDevice);

    cudaEvent_t s1,e1,s2,e2;
    cudaEventCreate(&s1);cudaEventCreate(&e1);
    cudaEventCreate(&s2);cudaEventCreate(&e2);

    cudaEventRecord(s1);
    int MAX_ITERS=n;
    for(int it=0;it<MAX_ITERS&&fsize>0;it++){
        cudaMemset(d_next_size,0,sizeof(int));
        int blocks=(fsize+255)/256;
        gpu_wcc_kernel<<<blocks,256>>>(d_row,d_col,d_comp,n,d_front,fsize,d_next,d_next_size);
        cudaDeviceSynchronize();
        cudaMemcpy(&fsize,d_next_size,sizeof(int),cudaMemcpyDeviceToHost);
        swap(d_front,d_next);
    }
    cudaEventRecord(e1);cudaEventSynchronize(e1);
    float static_time;cudaEventElapsedTime(&static_time,s1,e1);

    cudaMemcpy(comp.data(),d_comp,n*sizeof(int),cudaMemcpyDeviceToHost);

    unordered_map<int,int>mp;int static_max=0;
    for(int i=0;i<n;i++){mp[comp[i]]++;static_max=max(static_max,mp[comp[i]]);}
    cout<<"StaticLargest="<<static_max<<"\n";

    vector<int>comp_cpu(n);
    for(int i=0;i<n;i++)comp_cpu[i]=i;

    bool ch=true;
    auto cpu_s=chrono::high_resolution_clock::now();
    while(ch){
        ch=false;
        for(auto&e:edges_after){
            int u=e.first,v=e.second;
            int cu=comp_cpu[u],cv=comp_cpu[v];
            int m=min(cu,cv);
            if(cu>m){comp_cpu[u]=m;ch=true;}
            if(cv>m){comp_cpu[v]=m;ch=true;}
        }
        for(int i=0;i<n;i++)while(comp_cpu[i]!=comp_cpu[comp_cpu[i]])comp_cpu[i]=comp_cpu[comp_cpu[i]];
    }
    auto cpu_e=chrono::high_resolution_clock::now();
    double cpu_time=chrono::duration<double,chrono::milliseconds::period>(cpu_e-cpu_s).count();

    unordered_map<int,int>cpu_mp;int cpu_max=0;
    for(int i=0;i<n;i++){cpu_mp[comp_cpu[i]]++;cpu_max=max(cpu_max,cpu_mp[comp_cpu[i]]);}
    cout<<"CpuLargest="<<cpu_max<<"\n";

    vector<int>row_ptr1,col_idx1;
    buildCSR(n,edges_after,row_ptr1,col_idx1);

    cudaMemcpy(d_row,row_ptr1.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,col_idx1.data(),col_idx1.size()*sizeof(int),cudaMemcpyHostToDevice);

    vector<int>comp_dyn(n);
    for(int i=0;i<n;i++)comp_dyn[i]=comp[i];
    cudaMemcpy(d_comp,comp_dyn.data(),n*sizeof(int),cudaMemcpyHostToDevice);

    vector<int>dyn_front;
    dyn_front.reserve(updates.size());
    for(auto&e:updates)dyn_front.push_back(e.second);
    fsize=dyn_front.size();
    cudaMemcpy(d_front,dyn_front.data(),fsize*sizeof(int),cudaMemcpyHostToDevice);

    cudaEventRecord(s2);
    for(int it=0;it<MAX_ITERS&&fsize>0;it++){
        cudaMemset(d_next_size,0,sizeof(int));
        int blocks=(fsize+255)/256;
        gpu_wcc_kernel<<<blocks,256>>>(d_row,d_col,d_comp,n,d_front,fsize,d_next,d_next_size);
        cudaDeviceSynchronize();
        cudaMemcpy(&fsize,d_next_size,sizeof(int),cudaMemcpyDeviceToHost);
        swap(d_front,d_next);
    }
    cudaEventRecord(e2);cudaEventSynchronize(e2);
    float dynamic_time;cudaEventElapsedTime(&dynamic_time,s2,e2);

    cudaMemcpy(comp_dyn.data(),d_comp,n*sizeof(int),cudaMemcpyDeviceToHost);
    for(int i=0;i<n;i++)while(comp_dyn[i]!=comp_dyn[comp_dyn[i]])comp_dyn[i]=comp_dyn[comp_dyn[i]];

    unordered_map<int,int>mpd;int dyn_max=0;
    for(int i=0;i<n;i++){mpd[comp_dyn[i]]++;dyn_max=max(dyn_max,mpd[comp_dyn[i]]);}

    cout<<"DynamicLargest="<<dyn_max<<"\n";
    cout<<"DynamicComp="<<mpd.size()<<"\n";
    cout<<"StaticTime="<<static_time<<"ms\n";
    cout<<"DynamicTime="<<dynamic_time<<"ms\n";
    cout<<"Speedup="<<static_time/dynamic_time<<"x\n";

    cudaFree(d_row);cudaFree(d_col);cudaFree(d_comp);cudaFree(d_front);cudaFree(d_next);cudaFree(d_next_size);
    cudaDeviceReset();
    return 0;
}
