#include "RL4FS/RL4FS.h"

namespace RL4FS {

    DataSet* clearProcessesfortestdata()
    {
        if (Property::getProperty()->getDistributedNodes() > 1)
        {
#ifdef USE_MPICH
            int process_id;
            MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
            auto numstestdatasets = Property::getProperty()->getNumTempDatasets();
            if (numstestdatasets == 1)
            {
                MPI_Finalize();
                Property::getProperty()->setDistributedNodes(1);
                if (process_id > 0)
                {
                    LOG(INFO) << "process_id: " << process_id << ", clearProcesses: MPI_Finalize!";
                    return nullptr;
                }
            }
#else
            LOG(ERROR) << "clearProcesses Error: USE_MPICH is not defined!";
            exit(-1);
#endif // 
        }
        DataSet* testdataset = nullptr;
        if (RL4FS::Property::getProperty()->getTestDataPath() != "")
        {
            auto targetname = RL4FS::Property::getProperty()->getClassName();
            auto discretename = RL4FS::Property::getProperty()->getDiscreteFeatureName();
            auto datename = RL4FS::Property::getProperty()->getDateFeatureName();
            auto classnum = RL4FS::Property::getProperty()->getTargetClasses();
            auto datasetname = RL4FS::Property::getProperty()->getDatasetName();
            auto testdatapath = RL4FS::Property::getProperty()->getRootPath() + RL4FS::Property::getProperty()->getTestDataPath();
            testdataset = RL4FS::Load::loadData(testdatapath, targetname, discretename, datename, classnum, datasetname + "_test", true, true, true, true);
        }
        return testdataset;
	}

    DataSet* getValData()
    {
        DataSet* valdataset = nullptr;
        if (RL4FS::Property::getProperty()->getValDataPath() != "")
        {
            auto targetname = RL4FS::Property::getProperty()->getClassName();
            auto discretename = RL4FS::Property::getProperty()->getDiscreteFeatureName();
            auto datename = RL4FS::Property::getProperty()->getDateFeatureName();
            auto classnum = RL4FS::Property::getProperty()->getTargetClasses();
            auto datasetname = RL4FS::Property::getProperty()->getDatasetName();
            auto valdatapath = RL4FS::Property::getProperty()->getRootPath() + RL4FS::Property::getProperty()->getValDataPath();
            if (Property::getProperty()->getIsAllNumber())
                valdataset = RL4FS::Load::loadDataAllNumeber(valdatapath, targetname, discretename, datename, classnum, datasetname + "_val", true, false, true, true);
            else valdataset = RL4FS::Load::loadData(valdatapath, targetname, discretename, datename, classnum, datasetname + "_val", true, false, true, true);
        }
        return valdataset;
    }

    DataSet* LoadTrainData()
    {
        string datapath = RL4FS::Property::getProperty()->getRootPath() + RL4FS::Property::getProperty()->getDatasetPath();
        string targetname = RL4FS::Property::getProperty()->getClassName();
        string datasetname = RL4FS::Property::getProperty()->getDatasetName();
        std::vector<string> discretename = RL4FS::Property::getProperty()->getDiscreteFeatureName();
        std::vector<string> datename = RL4FS::Property::getProperty()->getDateFeatureName();
        int classnum = RL4FS::Property::getProperty()->getTargetClasses();
        DataSet* dataset = nullptr;
        if (Property::getProperty()->getIsAllNumber())
            dataset = RL4FS::Load::loadDataAllNumeber(datapath, targetname, discretename, datename, classnum, datasetname, true, true, true);
        else dataset = RL4FS::Load::loadData(datapath, targetname, discretename, datename, classnum, datasetname, true, true, true);
        auto valdataset = getValData();
        dataset->megerDataSet(valdataset);
        if (valdataset != nullptr) delete valdataset, valdataset = nullptr;
        return dataset;
    }

    DataSet* startFeatureSelection(DataSet* dataset, DataSet* &testdataset)
    {
        if (dataset == nullptr)
        {
            LOG(ERROR) << "startFeatureSelection Error: dataset is nullptr!";
            return nullptr;
        }
        RL4FS::StopWatch stop;
        stop.Start();
        LOG(INFO) << dataset->getName() << " Begin RL4FS!";
        std::vector<int> selectedfeatures;
        if(Property::getProperty()->getDatasetType() == DataType::CSV)
        {
            Params params(Property::getProperty()->getParams());
            ReinforcementLearning rl(dataset, &params);
            selectedfeatures = rl.optimize();
            LOG(INFO) << "RL4FS Finished! and  the best score is " << rl.getBestLoss() << ", the best status is : " << selectedfeatures;
        }
        else
        {
            FeatureSelection fs;
            fs.run(dataset, selectedfeatures);
            sort(selectedfeatures.begin(), selectedfeatures.end());
            LOG(INFO) << "RL4FS Finished! " << " the featureindex selected is : " << selectedfeatures;
        }
        stop.Stop();
        LOG(INFO) << "startFeatureSelection cost :" << std::to_string(stop.Elapsed()) << " us.";
        dataset->setSelectFeatures(selectedfeatures);
        dataset->updateDataSet();
        string savepath = RL4FS::Property::getProperty()->getRootPath() + RL4FS::Property::getProperty()->getSavePath();
        auto memorysize = dataset->getMemorySize();
        if ((memorysize >> 20) > Property::getProperty()->getLimitedMemory())
        {
            dataset->write(savepath);
            delete dataset;
            dataset = nullptr;
        }
        testdataset = clearProcessesfortestdata(); // load test data
        if (testdataset != nullptr)
        {
            testdataset->setSelectFeatures(selectedfeatures);
            testdataset->updateDataSet();
        }
        return dataset;
    }

    void saveData(DataSet*& dataset)
    {
        string savepath = RL4FS::Property::getProperty()->getRootPath() + RL4FS::Property::getProperty()->getSavePath();
        if (dataset != nullptr) { dataset->write(savepath); delete dataset; dataset = nullptr; }
    }

    void InitLog()
    {
        string logpath = RL4FS::Property::getProperty()->getRootPath() + RL4FS::Property::getProperty()->getLoggerPath();
        // if logpath is not exist, create it
        string logdir = logpath.substr(0, logpath.find_last_of("/"));
        if (logdir != "" && !std::filesystem::exists(logdir))
        {
            std::filesystem::create_directories(logdir);
            LOG(INFO) << "Create log directory: " << logdir;
        }
        el::Configurations defaultConf;
        // set path to log file
        defaultConf.set(el::Level::Global,
            el::ConfigurationType::Filename, logpath);
        el::Level level = static_cast<el::Level>(RL4FS::Property::getProperty()->getLoggerLevel());
        el::Loggers::reconfigureLogger("default", defaultConf);
        el::Loggers::addFlag(el::LoggingFlag::HierarchicalLogging);
        el::Loggers::setLoggingLevel(level);
        defaultConf.setToDefault();
    }

}//RL4FS



