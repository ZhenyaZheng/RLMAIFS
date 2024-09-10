#pragma once
#include "util/ClearClass.h"
#include "Load.h"
#include "util/MyException.h"
#include "LightGBM/c_api.h"
#include "rl/rl.h"
#include "FeatureSelection.h"
namespace RL4FS {
	
	DataSet* clearProcessesfortestdata();
	DataSet* startFeatureSelection(DataSet* dataset, DataSet* &testdataset);
	DataSet* getValData();
	DataSet* LoadTrainData();
	void saveData(DataSet*& dataset);
	void InitLog();
}//RL4FS
