# Graph-analytics-on-GPU
This repository contains CUDA implementations of fundamental graph algorithms optimized for parallel execution on NVIDIA GPUs.

##  Contents

| File | Description |
|------|--------------|
| `bfs.cu` | Breadth-First Search (BFS) using CUDA. |
| `pagerank.cu` | PageRank algorithm using iterative GPU computation. |
| `SCC.cu` | Strongly Connected Components detection using parallel graph traversal. |
| `dynamic_bfs.cu` | Dynamic BFS supporting edge/node insertions and deletions on GPU. |
| `dynamic_page_rank.cu` | Dynamic PageRank computation updating scores incrementally after graph changes. |
| `dynamic_sssp.cu` | Single Source Shortest Path caluclated dynamically after adding and modifying edges  |
| `wcc.cu` | Dynamic updation of components with incrementation of edges|

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
nvcc  pagerank.cu -o pagerank
```

### Compile SCC
```bash
nvcc  SCC.cu -o scc
```
### Compile Dynamic BFS

```bash
nvcc  dynamic_bfs.cu -o dynamicbfs
```

### Compile Dynamic PageRank
```bash
nvcc  dynamic_page_rank.cu -o dynamic_page_rank
```

### Compile Dynamic SSSp
```bash
nvcc  dynamic_sssp.cu -o dynamic_sssp
```

### Compile Dynamic WCC
```bash
nvcc  wcc.cu -o wcc
```
###  Running the Programs

After compiling, run the executables as follows:

```bash
./bfs
./pagerank
./scc
./dynamic_bfs
./dynamic_page_rank
./dynamic_sssp
./wcc
```
### Datasets used
This project uses real-world graphs from multiple domains, all taken from the SNAP dataset collection (Stanford Network Analysis Project):

1. Social Networks:
ego-Epinions1,
soc-Slashdot0922

3. Communication Networks:
email-EuAll

4. Web Graphs:
web-NotreDame

4.Road Networks:
roadNet-CA



All datasets can be downloaded from:
 https://snap.stanford.edu/data/index.html

Each dataset is provided as an edge-list text file, which can be directly used as input to the CUDA programs in this repository.

Developed by B.L.Kalyan and K.Balakrishna  
For project work on **Dynamic Graph Analytics on GPU**
