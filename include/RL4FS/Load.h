#pragma once
#include "./DataSet.h"
#include "csv2/reader.hpp"
#include "./util/Tools.h"
namespace RL4FS
{
	class Load
	{
		
	public:

		static bool StdDataSetIndex(DataSet* dataset, PFeatureInfo& featureinfo, int index);

		static bool DistributedStdDataSetIndex(DataSet* dataset, PFeatureInfo& featureinfo, bool ischange=false);

		static bool StdDataSet(DataSet* dataset, bool ischange = true, int index = -1);

		static bool DistributedStdDataSet(DataSet* dataset, bool ischange = true, int index = -1);

		static DataSet* loadData(string datapath, string classname = Property::getProperty()->getClassName(), std::vector<string> discretefeaturename = Property::getProperty()->getDiscreteFeatureName(), std::vector<string> datefeaturename = Property::getProperty()->getDateFeatureName(), int classes = Property::getProperty()->getTargetClasses(), string datasetname = Property::getProperty()->getDatasetName(), bool hastitle = true, bool ischange = true, bool loadtraindata = false, bool testdata = false);

		static DataSet* loadDataAllNumeber(string datapath, string classname = Property::getProperty()->getClassName(), std::vector<string> discretefeaturename = Property::getProperty()->getDiscreteFeatureName(), std::vector<string> datefeaturename = Property::getProperty()->getDateFeatureName(), int classes = Property::getProperty()->getTargetClasses(), string datasetname = Property::getProperty()->getDatasetName(), bool hastitle = true, bool ischange = true, bool loadtraindata = false, bool testdata = false);

	};


}//RL4FS
