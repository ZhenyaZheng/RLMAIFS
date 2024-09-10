# RLMAIFS: Reinforcement Learning-based Multi-Agent Interaction for Feature Selection

RLMAIFS is an efficient automatic feature selection method implemented in `C++17`, supporting large datasets (beyond single machine memory). Thanks to the work of [ThunderSVM](https://github.com/Xtra-Computing/thundersvm), RLMAIFS also supports sparse datasets ([libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)). Additionally, thanks to open-source tools like [csv2](https://github.com/p-ranav/csv2), [LightGBM](https://github.com/microsoft/LightGBM), [easyloggingpp](https://github.com/abumq/easyloggingpp), and [json](https://github.com/nlohmann/json).

[Chinese](README_zh.md)

## 1. **Installation:**

- ### **Windows:**

VisualStudio >= 2019

cmake >= 3.13

MPI >= 1.0.3 (optional)

[LightGBM](https://github.com/microsoft/LightGBM)

Execution steps:

First, compile LightGBM:

```shell
git submodule init
git submodule update
cd dep/LightGBM
git submodule init
git submodule update
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="./"
```

Then, select the `LightGBM.sln` project file, open it with VS, and click INSTALL in the solution. Then choose Build -> Build INSTALL from the menu bar.

Next, compile RLMAIFS:

```shell
cd <Project_DIR>
mkdir build
cd build
cmake ..  -DCMAKE_INSTALL_PREFIX="./" -DUSE_MPICH=ON -D_LIGHTGBM_INCLUDE_DIRS="dep/LightGBM/build/include/" -D_LIGHTGBM_LIBRARIES="dep/LightGBM/build/lib"
```

If you don't need distributed execution, you can omit `-DUSE_MPICH=ON`. If your MPI is not in the system directory, you need to specify `-DMPICH_INCLUDE_DIR` and `-DMPI_LIBRARY`. Then select the `RLMAIFS.sln` project file, open it with VS, and click INSTALL in the solution. Choose Build -> Build INSTALL from the menu to install `RLMAIFS.dll` and `RLMAIFSMain.exe` into the folder specified by `CMAKE_INSTALL_PREFIX`.

Explanation of cmake parameters:

`USE_MPICH`: Whether to use `MPI`

`MPICH_INCLUDE_DIR`: The include directory of `MPI`

`MPI_LIBRARY`: The library directory of `MPI`

`_LIGHTGBM_INCLUDE_DIRS`: The include directory of LightGBM

`_LIGHTGBM_LIBRARIES`: The library directory of LightGBM

- ### **Linux & MacOS:**

cmake >= 3.13
MPI >= 4.1 (optional)

[LightGBM](https://github.com/microsoft/LightGBM)

Execution steps:

First, compile LightGBM:

```shell
git submodule init
git submodule update
cd dep/LightGBM
git submodule init
git submodule update
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="./"
make install -j8
```

Then, compile RLMAIFS:

```shell
cd <Project_DIR>
mkdir build  
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$ZZROOT -DUSE_MPICH=ON -D_LIGHTGBM_INCLUDE_DIRS="dep/LightGBM/build/include/" -D_LIGHTGBM_LIBRARIES="dep/LightGBM/build/lib"
make install -j8
```

Similarly, if you don't need distributed execution, you can omit `-DUSE_MPICH=ON`. If your MPI is not in the system directory, you need to specify `-DMPICH_INCLUDE_DIR` and `-DMPI_LIBRARY`.

## 2. **Usage:**

- First, configure the file [property.json](config/property.json) in the config folder. You can create different json files for different datasets.

  Parameter description:

  1. **`rootpath`**: The root path of the project
  2. **`datasetpath`**: The path to the dataset, relative to **`rootpath`**
  3. **`savepath`**: The save path of the dataset, relative to **`rootpath`**
  4. **`classname`**: The name of the class feature, not required for sparse datasets
  5. **`datasetname`**: The name of the dataset
  6. `distributednodes`: Number of distributed nodes
  7. `datefeaturename`: Name of the time-series feature
  8. **`discretefeaturename`**: Name of the discrete feature
  9. `loggerpath`: Path for logs, relative to **`rootpath`**
  10. **`testdatapath`**: Path for the test dataset, relative to **`rootpath`**, if not available, leave as “”
  11. `wethreadnum`: Number of threads for parallel feature selection by multiple agents
  12. **`targetclasses`**: Number of class categories, for regression datasets, it is 0
  13. **`maxnumsfeatures`**: Number of features to retain
  14. **`datasettype`**: Type of the dataset: 0 - csv, 1 - distributed, 2 - libsvm classification, 3 - libsvm regression, 4 - chunked dataset
  15. `loggerlevel`: Log level, `1: Global, 2: Trace, 4: Debug, 8: Fatal, 16: Error, 32: Warning, 64: Verbose, 128: Info, 1024: Unknown`
  16. `missval`: Missing value in the dataset
  17. **`featurenum`**: Number of original features in the dataset
  18. `otherdatasethashead`: Whether non-first chunks of a large chunked dataset contain headers, only required for single-machine chunking
  19. `targetclassindex`: Index of the class feature, -1 for the last column, otherwise use positive integers, starting from 0
  20. `targetmutil`: Whether the class is multi-class
  21. `temppath`: Temporary path, relative to **`rootpath`**

**Please note**:

- Parameters in bold are those you might need to modify.

- RLMAIFS supports the following dataset types:

  1. Single csv dataset: Set `rootpath` + `datasetpath` to the absolute path of the dataset and `datasettype` to 0.
  2. Distributed csv dataset: Set `distributednodes`, `datasettype` to 1. Distribute the dataset across the nodes in the cluster and append an ID starting from 1 to each dataset based on the order in the node configuration file. For example, with four nodes, the dataset on the first node should append '0' to the name, and subsequent nodes append '1', '2', '3'...
  3. Single libsvm classification dataset: Set `rootpath` + `datasetpath` to the absolute path of the dataset and `datasettype` to 2.
  4. Single libsvm regression dataset: Set `rootpath` + `datasetpath` to the absolute path of the dataset and `datasettype` to 3.
  5. Distributed libsvm classification dataset: Set `distributednodes`, and set `rootpath` + `datasetpath` to the dataset's absolute path, with `datasettype` set to 2. Distribute the dataset as described above for csv datasets.
  6. Distributed libsvm regression dataset: Set `distributednodes`, and set `rootpath` + `datasetpath` to the dataset's absolute path, with `datasettype` set to 3. Distribute the dataset as described above for csv datasets.

- Execution commands:

  1. Distributed:

     ```shell
     Windows:
     mpiexec -np 2 --hostfile config ./RLMAIFSMain.exe ../config/property.json
     Linux & MacOS:
     mpirun -n 2 --hostfile config ./RLMAIFSMain ../config/property.json
     ```

     Example `config` file:

     ```shell
     192.168.0.1 slots=1
     192.168.0.2 slots=1
     ```

  2. Non-distributed:

     ```shell
     Windows:
     ./RLMAIFSMain.exe ../config/property.json
     Linux & MacOS:
     ./RLMAIFSMain ../config/property.json
     ```

