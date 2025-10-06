#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <chrono>
#include <vector>
#include <queue>

__global__ void random_graph_initializer(int *src,int *dest,int numEdges,int numNodes,unsigned int seed){
 int threadId=blockIdx.x*blockDim.x + threadIdx.x;
 if(threadId<numEdges){
    curandState state;
    curand_init(seed,threadId,0,&state);
    src[threadId]=curand(&state)%numNodes;
    dest[threadId]=curand(&state)%numNodes;
 }   
}

__global__ void countEdges(int *src,int *rowPtr,int numEdges){
 int threadId=blockIdx.x*blockDim.x + threadIdx.x;
 if(threadId<numEdges){
    atomicAdd(&rowPtr[src[threadId]+1],1);
 }
}

__global__ void bfs_kernel(int *rowPtr,int *colIdx,int *visited,int *frontier,int *next_frontier,int numNodes,int *done){
 int threadId=blockIdx.x*blockDim.x + threadIdx.x;
 if(threadId<numNodes && frontier[threadId]){
    frontier[threadId]=0;
    int st=rowPtr[threadId];
    int end=rowPtr[threadId+1];
    for(int i=st;i<end;i++){
        int v=colIdx[i];
        if(!visited[v]){
            if(atomicCAS(&visited[v],0,1)==0){
                next_frontier[v]=1;
                *done=0;
            }
        }
    }
 }
}

void bfs_cpu(std::vector<int>&rowPtr,std::vector<int>&colIdx,std::vector<int>&visited,int start){
 std::queue<int>q;
 visited[start]=1;
 q.push(start);
 while(!q.empty()){
  int u=q.front();
  q.pop();
  for(int i=rowPtr[u];i<rowPtr[u+1];i++){
   int v=colIdx[i];
   if(!visited[v]){
    visited[v]=1;
    q.push(v);
   }
  }
 }
}

int main(){
 srand((unsigned)time(NULL));
 unsigned int seed=(unsigned)time(NULL);
 int numNodes=10000;
 int numEdges=50000;
 printf("Random graph: %d nodes, %d edges\n",numNodes,numEdges);

 int *d_src,*d_dest,*d_rowPtr,*d_colIdx;
 cudaMalloc(&d_src,numEdges*sizeof(int));
 cudaMalloc(&d_dest,numEdges*sizeof(int));
 cudaMalloc(&d_rowPtr,(numNodes+1)*sizeof(int));
 cudaMalloc(&d_colIdx,numEdges*sizeof(int));
 cudaMemset(d_rowPtr,0,(numNodes+1)*sizeof(int));

 int threads=256;
 int blocks=(numEdges+threads-1)/threads;

 random_graph_initializer<<<blocks,threads>>>(d_src,d_dest,numEdges,numNodes,seed);
 cudaDeviceSynchronize();
 countEdges<<<blocks,threads>>>(d_src,d_rowPtr,numEdges);
 cudaDeviceSynchronize();

 int *h_rowPtr=(int*)malloc((numNodes+1)*sizeof(int));
 cudaMemcpy(h_rowPtr,d_rowPtr,(numNodes+1)*sizeof(int),cudaMemcpyDeviceToHost);
 for(int i=1;i<=numNodes;i++){
    h_rowPtr[i]+=h_rowPtr[i-1];
 }
 cudaMemcpy(d_rowPtr,h_rowPtr,(numNodes+1)*sizeof(int),cudaMemcpyHostToDevice);

 int *h_src=(int*)malloc(numEdges*sizeof(int));
 int *h_dest=(int*)malloc(numEdges*sizeof(int));
 cudaMemcpy(h_src,d_src,numEdges*sizeof(int),cudaMemcpyDeviceToHost);
 cudaMemcpy(h_dest,d_dest,numEdges*sizeof(int),cudaMemcpyDeviceToHost);

 int *h_colIdx=(int*)malloc(numEdges*sizeof(int));
 int *pos=(int*)calloc(numNodes,sizeof(int));
 for(int i=0;i<numEdges;i++){
    int u=h_src[i];
    int idx=h_rowPtr[u]+pos[u]++;
    h_colIdx[idx]=h_dest[i];
 }
 cudaMemcpy(d_colIdx,h_colIdx,numEdges*sizeof(int),cudaMemcpyHostToDevice);

 std::vector<int>rowPtr(h_rowPtr,h_rowPtr+numNodes+1);
 std::vector<int>colIdx(h_colIdx,h_colIdx+numEdges);
 std::vector<int>visited_cpu(numNodes,0);

 auto start_cpu=std::chrono::high_resolution_clock::now();
 bfs_cpu(rowPtr,colIdx,visited_cpu,0);
 auto end_cpu=std::chrono::high_resolution_clock::now();
 double cpu_ms=std::chrono::duration<double,std::milli>(end_cpu-start_cpu).count();

 int *d_visited,*d_frontier,*d_next_frontier,*d_done;
 cudaMalloc(&d_visited,numNodes*sizeof(int));
 cudaMalloc(&d_frontier,numNodes*sizeof(int));
 cudaMalloc(&d_next_frontier,numNodes*sizeof(int));
 cudaMalloc(&d_done,sizeof(int));
 cudaMemset(d_visited,0,numNodes*sizeof(int));
 cudaMemset(d_frontier,0,numNodes*sizeof(int));
 cudaMemset(d_next_frontier,0,numNodes*sizeof(int));
 int one=1;
 cudaMemcpy(d_visited,&one,sizeof(int),cudaMemcpyHostToDevice);
 cudaMemcpy(d_frontier,&one,sizeof(int),cudaMemcpyHostToDevice);

 threads=256;
 blocks=(numNodes+threads-1)/threads;
 int h_done;

 cudaEvent_t start,stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);
 cudaEventRecord(start);
 do{
  h_done=1;
  cudaMemcpy(d_done,&h_done,sizeof(int),cudaMemcpyHostToDevice);
  bfs_kernel<<<blocks,threads>>>(d_rowPtr,d_colIdx,d_visited,d_frontier,d_next_frontier,numNodes,d_done);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_done,d_done,sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(d_frontier,d_next_frontier,numNodes*sizeof(int),cudaMemcpyDeviceToDevice);
  cudaMemset(d_next_frontier,0,numNodes*sizeof(int));
 }while(h_done==0);
 cudaEventRecord(stop);
 cudaEventSynchronize(stop);
 float gpu_ms=0;
 cudaEventElapsedTime(&gpu_ms,start,stop);

 printf("CPU BFS time: %.2f ms\n",cpu_ms);
 printf("GPU BFS time: %.2f ms\n",gpu_ms);
 printf("Speedup: %.2fx\n",cpu_ms/gpu_ms);

 free(h_rowPtr);
 free(h_src);
 free(h_dest);
 free(h_colIdx);
 free(pos);
 cudaFree(d_src);
 cudaFree(d_dest);
 cudaFree(d_rowPtr);
 cudaFree(d_colIdx);
 cudaFree(d_visited);
 cudaFree(d_frontier);
 cudaFree(d_next_frontier);
 cudaFree(d_done);
 return 0;
}
