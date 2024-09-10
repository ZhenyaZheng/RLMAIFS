#include "RL4FS/RL4FS.h"

int main(int argc, char* argv[])
{
	//_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	string propertypath = "../config/property.json";
	if (argc > 1)
		propertypath = argv[1];
	else
	{
		if (propertypath.empty())
		{
			LOG(ERROR) << "Please input the property path!";
			exit(-1);
		}
	}

	RL4FS::Property::getProperty()->readProperty(propertypath);
	RL4FS::StopWatch stop;
	RL4FS::InitLog();
	stop.Start();
	LOG(INFO) << "Begin Load Data!";
	auto dataset = RL4FS::LoadTrainData();
	LOG(INFO) << dataset->getName() << " Begin RL4FS!";
	stop.Stop();
	LOG(INFO) << "before startFeatureConstruction cost :" << std::to_string(stop.Elapsed()) << " us.";
	RL4FS::DataSet* testdataset = nullptr;

	int maxnumsfeatures = RL4FS::Property::getProperty()->getMaxNumsFeatures();
	dataset = RL4FS::startFeatureSelection(dataset, testdataset);
	RL4FS::saveData(dataset);
	RL4FS::saveData(testdataset);
	RL4FS::ClearClass::clear();
	return 0;
}
