# Graph-analytics-on-GPU
This repository contains CUDA implementations of fundamental graph algorithms optimized for parallel execution on NVIDIA GPUs.

##  Contents

| File | Description |
|------|--------------|
| `bfs.cu` | Breadth-First Search (BFS) using CUDA. |
| `pagerank.cu` | PageRank algorithm using iterative GPU computation. |
| `SCC.cu` | Strongly Connected Components detection using parallel graph traversal. |

---

##  Requirements

- NVIDIA GPU with CUDA Compute Capability ≥ 5.0  
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) ≥ 11.0  
- `nvcc` compiler (comes with CUDA)  
- Linux or Windows with WSL (preferred for compilation)

##  Compilation Instructions (No Makefile)

You can compile each `.cu` file manually using the `nvcc` compiler.
##  Note
- Adjust -arch=sm_60 according to your GPU architecture (e.g., sm_70, sm_75, or sm_86) below
### Compile BFS
```bash
nvcc -O2 -arch=sm_60 bfs.cu -o bfs
```

### Compile PageRank
```bash
nvcc -O2 -arch=sm_60 pagerank.cu -o pagerank
```

### Compile SCC
```bash
nvcc -O2 -arch=sm_60 SCC.cu -o scc
```
##  Running the Programs

After compiling, run the executables as follows:

```bash
./bfs
./pagerank
./scc
```


Developed by B.L.Kalyan and K.Balakrishna  
For project work on **Dynamic Graph Analytics on GPU**
