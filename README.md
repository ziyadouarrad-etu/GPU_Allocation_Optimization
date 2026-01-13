# 🚀 GPU Allocation Optimizer: Linear Programming Approach


## 📝 Overview

This project addresses a resource allocation problem common in high-performance computing (HPC) environments. The goal is to minimize the total operational cost of deploying various AI models (**BERT, ResNet-50, LSTM**) across different GPU architectures (**A100, V100, T4**) while strictly adhering to hardware availability and workload demand constraints.

The solution utilizes **Linear Programming (LP)** via the **PuLP** library to reach an optimal integer solution, ensuring maximum cost-efficiency for infrastructure management.

---

## 📂 Project Architecture

The project follows a modular "Data-Driven" engineering design, separating configuration, logic, and reporting.

```text
GPU_Optimization/
├── data/                 # Input parameters
│   ├── config.json       # Global metadata (Model/GPU names, T_factor)
│   ├── costs.csv         # 3x3 Operational cost matrix
│   ├── availability.csv  # Hardware time limits
│   └── demands.csv       # Model deployment targets
├── src/                  # Modular Source Code
│   ├── optimizer.py      # PuLP Linear Programming logic
│   └── visualizer.py     # Absolute-path reporting suite
├── Plots/                # Generated Insights (Auto-created)
├── main.py               # System Orchestrator
├── requirements.txt      # Dependency manifest
└── README.md             # Documentation

```

---

## 🔢 Mathematical Formulation

The problem is modeled as follows:

### 1. Decision Variables

Let $x_{i,j}$ be the integer number of units of **Model $i$** assigned to **GPU $j$**, where:

- $i \in \{\text{BERT, ResNet-50, LSTM}\}$
- $j \in \{\text{A100, V100, T4}\}$

### 2. Objective Function

Minimize the total cost :

$$\text{Minimize } Z = \sum_{i=1}^{3} \sum_{j=1}^{3} \text{Cost}_{i,j} \cdot x_{i,j}$$

### 3. Constraints

- **Resource Availability**: The time consumed on each GPU must not exceed its limit.

$$\sum_{i=1}^{3} x_{i,j} \cdot \text{Time}_{i,j} \le \text{Availability}_j, \quad \forall j$$

- **Workload Demand**: All model deployment targets must be met.

$$\sum_{j=1}^{3} x_{i,j} = \text{Demand}_i, \quad \forall i$$

- **Integrity**: $x_{i,j} \ge 0$ and $x_{i,j} \in \mathbb{Z}$.

---

## 🧠 Scientific Methodology

Following the study by R. Raushan (2024) and MLPerf benchmarks, the execution time matrix is derived from a standardized reference unit.

### 1. Workload Unit Definition

An "Order Unit" corresponds to a Fine-Tuning session of 10 epochs.

- Baseline: Training ResNet-50 on an Nvidia T4 takes 15 hours ($1.5h/\text{epoch} \times 10$).

### 2. Relative Speed Factors ($\alpha$)

Relative acceleration factors were derived from MLPerf and Dell EMC benchmarks.

## 📊 Key Results & Insights

Running the simulation generates a comprehensive suite of analytics in the `Plots/` folder:

- **Allocation Heatmap**: Visualizes the concentration of models on specific hardware.
- **GPU Utilization**: Tracks the "time-burden" on each GPU type to identify bottlenecks.
- **Cost Distribution**: Pie charts representing the budget impact per model and hardware category.
- **Efficiency Metrics**: Average cost per model type to evaluate economic performance.

---

## 🛠 Installation & Usage

1. **Clone and Setup**:

```bash
git clone https://github.com/ziyadouarrad-etu/GPU_Allocation_Optimization.git
cd GPU_Allocation_Optimization

```

2. **Install Dependencies**:

```bash
pip install -r requirements.txt

```

3. **Execute Optimizer**:

```bash
python main.py

```

## 🌐 Sources

- NVIDIA Corporation. (2020). NVIDIA A100 Tensor Core GPU Datasheet & MLPerf
  Benchmarks.
- Raushan, R. (2024). Training ResNet-50 from Scratch: Lessons Beyond the Model. Medium
  Engineering Blog.
- Dell EMC. (2019). Deep Learning Performance on T4 GPUs with MLPerf Benchmarks.
