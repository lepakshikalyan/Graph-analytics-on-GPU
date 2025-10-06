#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <vector>
#include <unordered_map>
using namespace std;

__global__ void random_graph_initializer(int *src,int *dest,int numEdges,int numNodes,unsigned int seed){
 int tid=blockIdx.x*blockDim.x+threadIdx.x;
 if(tid<numEdges){
    curandState state;
    curand_init(seed,tid,0,&state);
    src[tid]=curand(&state)%numNodes;
    dest[tid]=curand(&state)%numNodes;
 }
}

__device__ int dsu_find(int *parent,int x){
 while(true){
    int p=parent[x];
    if(p==x) return x;
    int gp=parent[p];
    if(gp==p) return p;
    atomicCAS(&parent[x],p,gp);
    x=parent[x];
 }
}

__global__ void dsu_union_kernel(int *src,int *dest,int *parent,int numEdges){
 int tid=blockIdx.x*blockDim.x+threadIdx.x;
 if(tid<numEdges){
    int u=src[tid];
    int v=dest[tid];
    int pu, pv;
    while(true){
        pu=dsu_find(parent,u);
        pv=dsu_find(parent,v);
        if(pu==pv) break;
        if(pu<pv){
            if(atomicCAS(&parent[pv],pv,pu)==pv) break;
        }else{
            if(atomicCAS(&parent[pu],pu,pv)==pu) break;
        }
    }
 }
}

__global__ void compress_paths(int *parent,int numNodes){
 int tid=blockIdx.x*blockDim.x+threadIdx.x;
 if(tid<numNodes){
    int p=dsu_find(parent,tid);
    parent[tid]=p;
 }
}

int main(){
 srand((unsigned)time(NULL));
 unsigned int seed=(unsigned)time(NULL);
 int numNodes=5+rand()%10;
 int numEdges=numNodes*(2+rand()%3);
 printf("Random graph: %d nodes, %d edges\n",numNodes,numEdges);

 int *d_src,*d_dest,*d_parent;
 cudaMalloc(&d_src,numEdges*sizeof(int));
 cudaMalloc(&d_dest,numEdges*sizeof(int));
 cudaMalloc(&d_parent,numNodes*sizeof(int));

 int threads=256;
 int blocks=(numEdges+threads-1)/threads;
 random_graph_initializer<<<blocks,threads>>>(d_src,d_dest,numEdges,numNodes,seed);
 cudaDeviceSynchronize();

 int *h_parent=(int*)malloc(numNodes*sizeof(int));
 for(int i=0;i<numNodes;i++) h_parent[i]=i;
 cudaMemcpy(d_parent,h_parent,numNodes*sizeof(int),cudaMemcpyHostToDevice);

 dsu_union_kernel<<<blocks,threads>>>(d_src,d_dest,d_parent,numEdges);
 cudaDeviceSynchronize();

 blocks=(numNodes+threads-1)/threads;
 compress_paths<<<blocks,threads>>>(d_parent,numNodes);
 cudaDeviceSynchronize();

 cudaMemcpy(h_parent,d_parent,numNodes*sizeof(int),cudaMemcpyDeviceToHost);

 
 unordered_map<int, vector<int>> comp;
 for(int i=0;i<numNodes;i++){
     comp[h_parent[i]].push_back(i);
 }

 printf("Connected Components:\n");
 int idx=1;
 for(auto &kv:comp){
     printf("Component %d: ",idx++);
     for(int node:kv.second) printf("%d ",node);
     printf("\n");
 }

 free(h_parent);
 cudaFree(d_src);
 cudaFree(d_dest);
 cudaFree(d_parent);
 return 0;
}
