# RLMAIFS: 基于多智能交互强化学习的特征选择

​      

RLMAIFS是一个用`C++17`实现，支持大数据集（超过单机内存）的高效自动特征选择方法，得益于[ThunderSVM](https://github.com/Xtra-Computing/thundersvm)的工作，RLMAIFS也支持稀疏数据集（[libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)）。另外， 感谢[csv2](https://github.com/p-ranav/csv2), [LightGBM](https://github.com/microsoft/LightGBM)， [easyloggingpp](https://github.com/abumq/easyloggingpp)， [json](https://github.com/nlohmann/json)开源工具。

[English](README.md)

## 1. **安装方法：**

- ### **Windows：**

VisualStudio >= 2019

cmake >= 3.13

MPI >= 1.0.3 (可选的)

[LightGBM](https://github.com/microsoft/LightGBM) 

执行命令：

请先编译LightGBM：

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

然后选择`LightGBM.sln`项目文件，用VS打开，在解决方案里点击INSTALL，然后选择菜单栏的生成->生成INSTALL。

然后编译RLMAIFS：

```shell
cd <Project_DIR>
mkdir build
cd build
cmake ..  -DCMAKE_INSTALL_PREFIX="./" -DUSE_MPICH=ON -D_LIGHTGBM_INCLUDE_DIRS="dep/LightGBM/build/include/" -D_LIGHTGBM_LIBRARIES="dep/LightGBM/build/lib"
```

如果你不需要分布式运行，可以不用设置`-DUSE_MPICH=ON`, 如果你的MPI不在系统目录，则需要指定 `-DMPICH_INCLUDE_DIR 和 -DMPI_LIBRARY`。然后选择`RLMAIFS.sln`项目文件，用VS打开，在解决方案里点击INSTALL，然后选择菜单栏的生成->生成INSTALL，即可以安装`RLMAIFS.dll`以及`RLMAIFSMain.exe`到`CMAKE_INSTALL_PREFIX`指定的文件夹。

cmake的各参数所表示的意义：

`USE_MPICH`：是否使用`MPI`

`MPICH_INCLUDE_DIR`：`MPI`的include目录

`MPI_LIBRARY`：`MPI`的lib目录

`_LIGHTGBM_INCLUDE_DIRS`：LightGBM的include目录

`_LIGHTGBM_LIBRARIES`：LightGBM的lib目录

- ### **Linux & MacOS：**

cmake >= 3.13
MPI >= 4.1 (可选的)

[LightGBM](https://github.com/microsoft/LightGBM) 

执行命令

请先编译LightGBM：

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

然后编译RLMAIFS:

```shell
cd <Project_DIR>
mkdir build  
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$ZZROOT -DUSE_MPICH=ON -D_LIGHTGBM_INCLUDE_DIRS="dep/LightGBM/build/include/" -D_LIGHTGBM_LIBRARIES="dep/LightGBM/build/lib"
make install -j8
```
同理，如果你不需要分布式运行，可以不用设置`-DUSE_MPICH=ON`, 如果你的MPI不在系统目录，则需要指定 `-DMPICH_INCLUDE_DIR 和 -DMPI_LIBRARY`。

## 2. **使用方法：**

- 首先对config里的文件[property.json](config/property.json)进行配置，你可以为不同的数据集创建不同的json：

  参数说明：

  1. **`rootpath`**：项目的根路径
  2. **`datasetpath`**：数据集的路径，相对于**`rootpath`**
  3. **`savepath`**：数据集保存的路径，相对于**`rootpath`**
  4. **`classname`**：class特征的名称，sparse数据集不需要
  5. **`datasetname`**：数据集的名字
  6. `distributednodes`：分布式节点个数
  8. `datefeaturename`：时间序列特征的名字
  9. **`discretefeaturename`**：离散型特征的名字
  10. `loggerpath`：日志的路径，相对于**`rootpath`**
  12. **`testdatapath`**：测试集路径，相对于**`rootpath`**，没有的话为“”
  14. `wethreadnum`：多智能体并行特征选择的线程个数
  15. **`targetclasses`**：class的类别数量，回归数据集为0
  16. **`maxnumsfeatures`**：保留的特征数量
  17. **`datasettype`**：数据集类型，0：csv数据，1：分布式数据，2：libsvm分类数据， 3：libsvm回归数据，4：分块数据集
  18. `loggerlevel`：日志等级，`1：Global， 2：Trace， 4：Debug， 8：Fatal， 16：Error， 32：Warning， 64： Verbose，128：Info，1024：Unknown`
  19. `missval`：数据集中的缺失值
  20. **`featurenum`**：数据集中原始特征的数量
  24. `otherdatasethashead`：分块大数据集的非首块是否含有标题行, 仅单机分块时需要设置
  25. `targetclassindex`：class特征的索引，-1代表最后一列，其他请用正数表示，从0开始计数。
  26. `targetmutil`：class是否为多类别
  27. `temppath`：temp路径，相对于**`rootpath`**

**请注意**：

- 以上标为粗体的参数都是可能需要改动的。
- RLMAIFS支持以下数据集的处理：
  1. 单个csv数据集：设置`rootpath`+`datasetpath`为数据集的绝对路径，`datasettype`设置为0。
  3. 分布式csv数据集：设置`distributednodes`，`datasettype`设置为1。把数据集分发到集群的节点中，根据节点配置文件的顺序在数据集后面添加标识ID，从1开始。例如一共有四个节点，每个节点设置`rootpath`+`datasetpath`为数据集的绝对路径，第一个节点上的数据集在磁盘上的名字需要在`rootpath`+`datasetpath`后加上‘0’，同理后面的节点依次加上‘1’, ‘2’，‘3’......
  4. 单个libsvm分类数据集：设置`rootpath`+`datasetpath`为数据集的绝对路径，`datasettype`设置为2。
  5. 单个libsvm回归数据集：设置`rootpath`+`datasetpath`为数据集的绝对路径，`datasettype`设置为3。
  8. 分布式libsvm分类数据集：设置`distributednodes`，设置`rootpath`+`datasetpath`为数据集的绝对路径，`datasettype`设置为2。把数据集分发到集群的节点中，根据节点配置文件的顺序在数据集后面添加标识ID，从1开始。例如一共有四个节点，每个节点设置`rootpath`+`datasetpath`为数据集的绝对路径，第一个节点上的数据集在磁盘上的名字需要在`rootpath`+`datasetpath`后加上‘0’，同理后面的节点依次加上‘1’ ,‘2’，‘3’......
  9. 分布式libsvm回归数据集：设置`distributednodes`，设置`rootpath`+`datasetpath`为数据集的绝对路径，`datasettype`设置为3。把数据集分发到集群的节点中，根据节点配置文件的顺序在数据集后面添加标识ID，从1开始。例如一共有四个节点，每个节点设置`rootpath`+`datasetpath`为数据集的绝对路径，第一个节点上的数据集在磁盘上的名字需要在`rootpath`+`datasetpath`后加上‘0’，同理后面的节点依次加上‘1’ ,‘2’，‘3’......
  
- 执行命令：

  1. 分布式

     ```
     Windows:
     mpiexec -np 2 --hostfile config ./RLMAIFSMain.exe ../config/property.json
     Linux & MacOS:
     mpirun -n 2 --hostfile config ./RLMAIFSMain ../config/property.json
     
     ```

     `config`文件例子：

     ```shell
     192.168.0.1 slots=1
     192.168.0.2 slots=1
     ```

  2. 非分布式

     ```shell
     Windows:
     ./RLMAIFSMain.exe ../config/property.json
     Linux & MacOS:
     ./RLMAIFSMain ../config/property.json
     ```

