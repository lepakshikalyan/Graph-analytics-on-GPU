#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <chrono>
#include <vector>

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

__global__ void pagerankKernel(int *rowPtr,int *colIdx,float *rankOld,float *rankNew,int numNodes,float damping){
 int threadId=blockIdx.x*blockDim.x + threadIdx.x;
 if(threadId<numNodes){
    float sum=0.0f;
    int st=rowPtr[threadId];
    int end=rowPtr[threadId+1];
    for(int i=st;i<end;i++){
        int v=colIdx[i];
        int outDegree=rowPtr[v+1]-rowPtr[v];
        if(outDegree>0){
            sum+=(rankOld[v]/outDegree);
        }
    }
    rankNew[threadId]=(1.0f-damping)/numNodes+(damping*sum);
 }
}

void pagerank_cpu(std::vector<int>&rowPtr,std::vector<int>&colIdx,std::vector<float>&rank,int numNodes,float damping,int max_iters){
 std::vector<float>newRank(numNodes,0.0f);
 for(int iter=0;iter<max_iters;iter++){
  for(int i=0;i<numNodes;i++){
   float sum=0.0f;
   for(int j=rowPtr[i];j<rowPtr[i+1];j++){
    int v=colIdx[j];
    int outdeg=rowPtr[v+1]-rowPtr[v];
    if(outdeg>0) sum+=rank[v]/outdeg;
   }
   newRank[i]=(1.0f-damping)/numNodes+damping*sum;
  }
  rank.swap(newRank);
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
    int v=h_dest[i];
    int idx=h_rowPtr[v]+pos[v]++;
    h_colIdx[idx]=h_src[i];
 }
 cudaMemcpy(d_colIdx,h_colIdx,numEdges*sizeof(int),cudaMemcpyHostToDevice);
 
 float *d_rankOld,*d_rankNew;
 cudaMalloc(&d_rankOld,numNodes*sizeof(float));
 cudaMalloc(&d_rankNew,numNodes*sizeof(float));
 
 float *h_rank=(float*)malloc(numNodes*sizeof(float));
 for(int i=0;i<numNodes;i++) h_rank[i]=1.0f/numNodes;
 cudaMemcpy(d_rankOld,h_rank,numNodes*sizeof(float),cudaMemcpyHostToDevice);
 
 threads=256;
 blocks=(numNodes+threads-1)/threads;
 float damping=0.85f;
 int max_iters=20;

 std::vector<int>rowPtr(h_rowPtr,h_rowPtr+numNodes+1);
 std::vector<int>colIdx(h_colIdx,h_colIdx+numEdges);
 std::vector<float>rank_cpu(numNodes,1.0f/numNodes);

 auto start_cpu=std::chrono::high_resolution_clock::now();
 pagerank_cpu(rowPtr,colIdx,rank_cpu,numNodes,damping,max_iters);
 auto end_cpu=std::chrono::high_resolution_clock::now();
 double cpu_ms=std::chrono::duration<double,std::milli>(end_cpu-start_cpu).count();

 cudaEvent_t start,stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);
 cudaEventRecord(start);
 for(int iter=0;iter<max_iters;iter++){
    pagerankKernel<<<blocks,threads>>>(d_rowPtr,d_colIdx,d_rankOld,d_rankNew,numNodes,damping);
    cudaDeviceSynchronize();
    float *tmp=d_rankOld;
    d_rankOld=d_rankNew;
    d_rankNew=tmp;
 }
 cudaEventRecord(stop);
 cudaEventSynchronize(stop);
 float gpu_ms=0;
 cudaEventElapsedTime(&gpu_ms,start,stop);

 cudaMemcpy(h_rank,d_rankOld,numNodes*sizeof(float),cudaMemcpyDeviceToHost);
 
 printf("CPU PR time: %.2f ms\n",cpu_ms);
 printf("GPU PR time: %.2f ms\n",gpu_ms);
 printf("Speedup: %.2fx\n",cpu_ms/gpu_ms);

 free(h_rowPtr);
 free(h_src);
 free(h_dest);
 free(h_colIdx);
 free(pos);
 free(h_rank);
 cudaFree(d_src);
 cudaFree(d_dest);
 cudaFree(d_rowPtr);
 cudaFree(d_colIdx);
 cudaFree(d_rankOld);
 cudaFree(d_rankNew);
 return 0;
}
