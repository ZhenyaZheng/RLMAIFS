#pragma once
#include "RL4FS/RL4FSStruct.h"
#include "nlohmann/json.hpp"
namespace RL4FS {
    class ClearClass;
    class Property
    {
    public:
        static Property* getProperty();
        static bool freeProperty();
        long long getLimitedMemory();
        Property* setLimitedMemory(long long limitedmemory);
        string getDatasetName();
        Property* setDatasetName(const string& datasetname);
        int getThreadNum();
        Property* setThreadNum(int threadnum);
        int getWeThreadNum();
        Property* setWeThreadNum(int wethreadnum);
        int getTargetClassIndex();
        Property* setTargetClassIndex(int targetclassindex);
        bool getTargetMutil();
        Property* setTargetMutil(bool targetmutil);
        int getTargetClasses();
        Property* setTargetClasses(int targetclasses);
        string getRootPath();
        Property* setRootPath(const string& rootpath);

        bool getOtherDatasetHasHead();
        Property* setOtherDatasetHasHead(bool otherdatasethashead);
        string getLoggerPath();
        Property* setLoggerPath(const string& loggerpath);
        string getTempPath();
        Property* setTempPath(const string& temppath);
        string getClassName();
        Property* setClassName(const string& classname);
        int getNumTempDatasets();
        Property* setNumTempDatasets(int numtempdatasets);
        std::vector<string> getDiscreteFeatureName();
        Property* setDiscreteFeatureName(const std::vector<string>& discretefeaturename);
        std::vector<string> getDateFeatureName();
        Property* setDateFeatureName(const std::vector<string>& datefeaturename);
        int getDistributedNodes();
        Property* setDistributedNodes(int distributednodes);
        DataType getDatasetType();
        Property* setDatasetType(DataType datasettype);
        string getTestDataPath();
        Property* setTestDataPath(const string& testdatapath);
        Property* readProperty(string propertypath);
        friend ClearClass;
        string getDatasetPath();
        Property* setDatasetPath(const string& datasetpath);
        string getSavePath();
        Property* setSavePath(const string& savepath);
        int getMaxNumsFeatures();
        Property* setMaxNumsFeatures(int maxnumsfeatures);
        int getFeatureNum();
        Property* setFeatureNum(int featurenum);
        int getLoggerLevel();
        Property* setLoggerLevel(int loggerlevel);
        string getMissVal();
        Property* setMissVal(const string& missval);
        string getValDataPath();
        Property* setValDataPath(const string& valdatapath);
        const std::unordered_map<string, string>& getLightGBMParams();
        Property* setLightGBMParams(const std::unordered_map<string, string>& lightgbmparams);
        bool getIsAllNumber();
        Property* setIsAllNumber(bool isallnumber);
        const std::unordered_map<string, string>& getParams();
        Property* setParams(const std::unordered_map<string, string>& params);
        ~Property();

    private:
        static Property* m_instance;
        long long m_limitedmemory = 20000;// the limited memory MB
        int m_threadnum = 10;// the number of threads for each operator
        int m_wethreadnum = 10;// the number of threads for generating features
        string m_datasetname = "dataset";// the name of dataset
        int m_targetclassindex = -1; // the index of target class
        bool m_targetmutil = false; // whether the target is mutil classes
        int m_targetclasses = 2; // the number of target classes
        string m_rootpath = ""; // the root path
        string m_loggerpath = "logs/";// the path of logs
        string m_classname = "classes";// the name of classes
        std::vector<string>  m_discretefeaturename = std::vector<string>(); // the name of discrete features
        std::vector<string>  m_datefeaturename = std::vector<string>(); // the name of date features
        string m_temppath = "temp/";// the path of temp files
        bool m_otherdatasethashead = false; // whether the other dataset has head
        int m_numtempdatasets = 2;// the number of temp datasets
        int m_distributednodes = 1;// the number of distributed nodes
        DataType m_datasettype = DataType::CSV;// 0:CSV 1:Distributed 2:LibSVMCF 3:LibSVMRG 4:BigData
        string m_testdatapath = "test";// the path of test data
        string m_datasetpath = "data/dataset/";// the path of dataset
        string m_savepath = "data/save/";// the path of save
        int m_maxnumsfeatures = 0;// the maximum number of features
        int m_featurenum = 0;// the number of features
        int m_loggerlevel = 0;// the level of logger
        string m_missval = "?";// the missing value
        string m_valdatapath = "";// the path of val data
        std::unordered_map<string, string> m_lightgbmparams{
            {"boosting_type", "gbdt"},
            {"num_leaves", "31"},
            {"learning_rate", "0.05"},
            {"feature_fraction", "0.9"},
            {"bagging_fraction", "0.8"},
            {"bagging_freq", "5"},
            {"verbosity", "0"},
            {"num_iterations", "1000"},
            {"min_data_in_leaf", "20"},
            {"lambda_l2", "0.001"},
            {"important_type", "1"}
            };
        bool m_isallnumber = true; // whether all features are numbers
        std::unordered_map<string, string> m_params{
            {"gamma", "0.5"},
            {"alpha", "0.1"},
            {"epsilon", "0.9"},
            {"n_search_iterations", "10"},
            {"state", ""},
            {"maxIterParticle", "10"},
            {"n_individuals", "10"},
            {"feature_size", "0"},
        }; // the parameters for reinforcement learning
        Property() {}
    };
}//RL4FS


