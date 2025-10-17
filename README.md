[![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![University of Florence](https://i.imgur.com/1NmBfH0.png)](https://ingegneria.unifi.it)

📄 **[Read the Full Report](https://github.com/mattiamarilli/SVMSMOParallelNoGIL/blob/main/report/SMO_Parallel_Report.pdf)**
🎥 **[Watch the Presentation](https://github.com/mattiamarilli/SVMSMOParallelNoGIL/blob/main/report/SMO_Parallel_Presentation.pdf)**


# Custom SMO SVM Parallel Implementation

This project implements a **custom Support Vector Machine (SVM)** classifier utilizing the **Sequential Minimal Optimization (SMO)** algorithm. Its core feature is the **multithreaded parallelization** of the **Radial Basis Function (RBF) kernel** column calculation, which is a key bottleneck in the SMO training phase.

The primary goal is to **benchmark the speedup and efficiency** achieved by increasing the number of threads for this critical CPU-bound operation, varying dataset size and feature count.

---


## 💡 Configuration and Context: Python 3.13 + GIL

The effectiveness of this multithreaded design is crucially dependent on Python's **Global Interpreter Lock (GIL)**.

### The GIL Bottleneck

In standard Python builds, the GIL prevents multiple threads from executing Python bytecode simultaneously, effectively nullifying the benefit of multithreading for **CPU-bound** tasks (like RBF kernel calculation) on multi-core systems.

### Python 3.13: Unlocking True Parallelism

The Python community has introduced the option to disable the GIL, available in **Python 3.13**.

> **To achieve genuine parallelism and accurately measure the maximum possible speedup ($S_P$) and efficiency ($E_P$) of this project's RBF kernel implementation, the benchmark must be executed using Python 3.13 compiled with the `--disable-gil` flag.**

This allows the `ThreadPoolExecutor` to execute the RBF distance computation in parallel across multiple CPU cores, validating the efficiency of the parallel code.

### Prerequisites for High-Performance Multithreading

1.  **Python 3.13 Source Code**: Download the Python 3.13 source tarball.
2.  **Compilation Flag**: Configure the build process with the `--disable-gil` option.

#### Compilation Steps (Recommended for Benchmarking):

```bash
# 1. Download and Extract Source
wget [https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz](https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz)
tar -xf Python-3.13.0.tgz
cd Python-3.13.0

# 2. Configure with GIL Disabled
./configure --disable-gil

# 3. Compile and Install (use 'altinstall' to avoid overwriting system python)
make
sudo make altinstall 
# Alternatively, specify a prefix: ./configure --disable-gil --prefix=$HOME/python3.13
```

#### How to run the script
```bash
PYTHON_GIL=0 python yournogilscript.py
```

## Overview

The `smoparallel.SVM` class provides an SVM implementation focused on high-performance kernel computation. The `benchmark_svm` script systematically tests this parallel implementation.

### Key Phases Under Benchmark:

1.  **Data Generation & Scaling**: Synthetic classification data created using `sklearn.datasets.make_classification`.
2.  **Training (Fit)**: The SMO algorithm, with time specifically measured for the parallel RBF column calculation (`sum_columns_calculation_time`).
3.  **Prediction (Predict)**: Testing the trained model on a hold-out set.

The performance of the **RBF column calculation** is tracked against the total number of threads utilized (`numthreads`).

---

## 🛠️ Key Features and Implementation

### `smoparallel.SVM` Class

| Feature | Description |
| :--- | :--- |
| **SMO Algorithm** | Implements the standard SMO procedure for optimizing Lagrange multipliers ($\alpha$). |
| **Kernel Support** | Supports **Linear** and **RBF** kernels. |
| **Parallel RBF** | **`rbf_kernel_column_multithread`** uses **`concurrent.futures.ThreadPoolExecutor`** to distribute the calculation of a kernel column across `self.numthreads`. |
| **Metrics Tracking** | Explicitly records the time spent in parallel kernel calculation (`self.sum_columns_calculation_time`). |

### Benchmark Configuration

The `benchmark_svm` function tests the following scenarios, averaging results over 10 runs (`runs=10`):

| Parameter | Values Tested |
| :--- | :--- |
| **Dataset Samples ($N$)** | `[3000]` |
| **Features ($D$)** | `[20, 50, 100, 200, 400, 600]` |
| **Thread Count ($P$)** | `[1, 2, 4, 8, 16]` |

---

## 📈 Performance Evaluation

The benchmark output in `svm_benchmark_report.txt` provides structured tables detailing the results for each dataset configuration.

### Core Metrics

The analysis focuses on the parallel kernel calculation time compared to the single-threaded baseline ($P=1$).

| Metric | Formula | Goal |
| :--- | :--- | :--- |
| **$T_{Columns}$** | `sum_columns_calculation_time` | Should decrease as $P$ increases. |
| **Speedup ($S_P$)** | $\frac{T_{Columns}(P=1)}{T_{Columns}(P)}$ | Should approach $P$. |
| **Efficiency ($E_P$)** | $\frac{S_P}{P} \times 100$ | Should be close to $100\%$ (perfect scaling). |
| **Accuracy** | Correct Predictions / Total Test Samples | Ensure model convergence is stable across threads. |

### Expected Results

* **RBF Kernel Parallelism**: For CPU-bound tasks like the RBF distance calculation, significant speedup is expected only when the **Global Interpreter Lock (GIL)** is disabled (see below).
* **Overhead**: Without GIL disabling, low feature counts may show *overhead* due to thread management costs overwhelming the limited parallelism.
* **Scaling**: Higher feature counts and a disabled GIL should demonstrate better *Speedup* and *Efficiency*.
