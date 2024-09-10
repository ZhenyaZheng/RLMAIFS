#ifndef THUNDERSVM_THUNDERSVM_H
#define THUNDERSVM_THUNDERSVM_H
#include <cstdlib>
#include "../util/log.h"
#include <string>
#include <vector>
#include "math.h"
#include "util/common.h"
using std::string;
using std::vector;

#ifdef USE_DOUBLE
typedef double kernel_type;
typedef double float_type;
#else
typedef float kernel_type;
typedef float float_type;
#endif
#endif //THUNDERSVM_THUNDERSVM_H
