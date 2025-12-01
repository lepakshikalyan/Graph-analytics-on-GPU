#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <chrono>
#include <vector>

using namespace std;

__global__ void random_graph_initializer(int *src,int *dest,int numEdges,int numNodes,unsigned int seed){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id<numEdges){
        curandState state;
        curand_init(seed,id,0,&state);
        src[id]=curand(&state)%numNodes;
        dest[id]=curand(&state)%numNodes;
    }
}

__global__ void countInDegree(int *dest,int *rowPtr,int numEdges){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id<numEdges){
        atomicAdd(&rowPtr[dest[id]+1],1);
    }
}

__global__ void countOutDegree(int *src,int *outdeg,int numEdges){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id<numEdges){
        atomicAdd(&outdeg[src[id]],1);
    }
}

__global__ void pagerankKernel(int *rowPtr,int *colIdx,int *outdeg,
                               float *rankOld,float *rankNew,
                               int numNodes,float damping)
{
    int v=blockIdx.x*blockDim.x+threadIdx.x;
    if(v<numNodes){
        float sum=0.0f;
        for(int i=rowPtr[v];i<rowPtr[v+1];i++){
            int u=colIdx[i];
            int od=outdeg[u];
            if(od>0) sum+=rankOld[u]/od;
        }
        rankNew[v]=(1.0f-damping)/numNodes + damping*sum;
    }
}

void pagerank_cpu(const vector<int>& rowPtr,const vector<int>& colIdx,
                  const vector<int>& outdeg,vector<float>& rank,
                  int numNodes,float damping,int iters)
{
    vector<float> newRank(numNodes);
    for(int it=0;it<iters;it++){
        for(int v=0;v<numNodes;v++){
            float sum=0.0f;
            for(int i=rowPtr[v];i<rowPtr[v+1];i++){
                int u=colIdx[i];
                int od=outdeg[u];
                if(od>0) sum+=rank[u]/od;
            }
            newRank[v]=(1.0f-damping)/numNodes + damping*sum;
        }
        rank.swap(newRank);
    }
}

int main(){
    srand(time(NULL));
    unsigned int seed=time(NULL);

    int numNodes=10000;
    int numEdges=50000;

    printf("Random graph: %d nodes, %d edges\n",numNodes,numEdges);

    int *d_src,*d_dest,*d_rowPtr,*d_colIdx,*d_outdeg;
    cudaMalloc(&d_src,numEdges*sizeof(int));
    cudaMalloc(&d_dest,numEdges*sizeof(int));
    cudaMalloc(&d_rowPtr,(numNodes+1)*sizeof(int));
    cudaMalloc(&d_colIdx,numEdges*sizeof(int));
    cudaMalloc(&d_outdeg,numNodes*sizeof(int));

    cudaMemset(d_rowPtr,0,(numNodes+1)*sizeof(int));
    cudaMemset(d_outdeg,0,numNodes*sizeof(int));

    int threads=256;
    int blocks=(numEdges+threads-1)/threads;

    random_graph_initializer<<<blocks,threads>>>(d_src,d_dest,numEdges,numNodes,seed);
    cudaDeviceSynchronize();

    countInDegree<<<blocks,threads>>>(d_dest,d_rowPtr,numEdges);
    countOutDegree<<<blocks,threads>>>(d_src,d_outdeg,numEdges);
    cudaDeviceSynchronize();

    int *h_rowPtr=(int*)malloc((numNodes+1)*sizeof(int));
    cudaMemcpy(h_rowPtr,d_rowPtr,(numNodes+1)*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=1;i<=numNodes;i++) h_rowPtr[i]+=h_rowPtr[i-1];
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

    int *h_outdeg=(int*)malloc(numNodes*sizeof(int));
    cudaMemcpy(h_outdeg,d_outdeg,numNodes*sizeof(int),cudaMemcpyDeviceToHost);

    float *d_rankOld,*d_rankNew;
    cudaMalloc(&d_rankOld,numNodes*sizeof(float));
    cudaMalloc(&d_rankNew,numNodes*sizeof(float));

    float *h_rank=(float*)malloc(numNodes*sizeof(float));
    for(int i=0;i<numNodes;i++) h_rank[i]=1.0f/numNodes;
    cudaMemcpy(d_rankOld,h_rank,numNodes*sizeof(float),cudaMemcpyHostToDevice);

    vector<int> rowPtr(h_rowPtr,h_rowPtr+numNodes+1);
    vector<int> colIdx(h_colIdx,h_colIdx+numEdges);
    vector<int> outdeg(h_outdeg,h_outdeg+numNodes);
    vector<float> rank_cpu(numNodes,1.0f/numNodes);

    float damping=0.85f;
    int max_iters=20;

    auto t1=chrono::high_resolution_clock::now();
    pagerank_cpu(rowPtr,colIdx,outdeg,rank_cpu,numNodes,damping,max_iters);
    auto t2=chrono::high_resolution_clock::now();
    double cpu_ms=chrono::duration<double,milli>(t2-t1).count();
    float sum=0;
    for(int i=0;i<numNodes;i++){
	sum+=rank_cpu[i];
    }
    
    cudaEvent_t g1,g2;
    cudaEventCreate(&g1);
    cudaEventCreate(&g2);
    cudaEventRecord(g1);

    blocks=(numNodes+threads-1)/threads;
    for(int it=0;it<max_iters;it++){
        pagerankKernel<<<blocks,threads>>>(d_rowPtr,d_colIdx,d_outdeg,d_rankOld,d_rankNew,numNodes,damping);
        cudaDeviceSynchronize();
        float *tmp=d_rankOld; d_rankOld=d_rankNew; d_rankNew=tmp;
    }

    cudaEventRecord(g2);
    cudaEventSynchronize(g2);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms,g1,g2);

    cudaMemcpy(h_rank,d_rankOld,numNodes*sizeof(float),cudaMemcpyDeviceToHost);
    float sum1=0;
    for(int i=0;i<numNodes;i++){
    	sum1+=h_rank[i];
    }
    printf("CPU PR time: %.2f ms\n",cpu_ms);
    printf("GPU PR time: %.2f ms\n",gpu_ms);
    printf("Speedup: %.2fx\n",cpu_ms/gpu_ms);
    printf("cpu ranks sum %.2f\n",sum);
    printf("gpu ranks sum %.2f\n",sum1);
    //cout<<"cpu ranks sum"<<sum;
    //cout<<"gpu ranks sum"<<sum1;
    free(h_rowPtr);
    free(h_src);
    free(h_dest);
    free(h_colIdx);
    free(pos);
    free(h_rank);
    free(h_outdeg);

    cudaFree(d_src);
    cudaFree(d_dest);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_outdeg);
    cudaFree(d_rankOld);
    cudaFree(d_rankNew);
}
