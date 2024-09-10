#pragma once
#include <string>
#include <iostream>
#include <set>
#include <map>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>

#include <random>
#include <queue>
#include <atomic>
#include <omp.h>
#include <thread>
#ifdef USE_MPICH
#include "mpi.h"
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif // !NOMINMAX
// #include <filesystem>
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(WIN32)
#include "util/mydirent.h"
#else
#include <dirent.h>
#endif
#include "util/log.h"



using std::string, std::cout, std::exception, std::endl, std::ios, std::ofstream, std::ifstream, std::istream, std::ostream ;
namespace RL4FS {
#define MAX_NUM_CATEGORY 500
#ifdef USE_DOUBLE
    using MyDataType = double;
#else
    using MyDataType = float;
#endif
    
    enum class FeatureType
    {
        Numeric, Discrete, Date, String
    };
    enum class OutType
    {
        Numeric, Discrete, Date, String
    };
    enum class DataType
    {
		CSV, Distribute, LibSVMCF, LibSVMRG, DIR
	};
    
}//RL4FS