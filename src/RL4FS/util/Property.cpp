#include "RL4FS/util/Property.h"
namespace RL4FS {
    
    Property* Property::getProperty()
    {
        if (m_instance == nullptr)
        {
            return m_instance = new Property();
        }
        else
            return m_instance;
    }
    bool Property::freeProperty()
    {
        delete m_instance;
        m_instance = nullptr;
        return true;
    }
    long long Property::getLimitedMemory()
    {
        return m_limitedmemory;
    }
    Property* Property::setLimitedMemory(long long limitedmemory)
    {
        m_limitedmemory = limitedmemory;
        return m_instance;
    }
    
    string Property::getDatasetName()
    {
        return m_datasetname;
    }
    Property* Property::setDatasetName(const string& datasetname)
    {
        m_datasetname = datasetname;
        return m_instance;
    }
    
    int Property::getThreadNum()
    {
        return m_threadnum;
    }
    Property* Property::setThreadNum(int threadnum)
    {
        m_threadnum = threadnum;
        return m_instance;
    }
    int Property::getWeThreadNum()
    {
        return m_wethreadnum;
    }
    Property* Property::setWeThreadNum(int wethreadnum)
    {
        m_wethreadnum = wethreadnum;
        return m_instance;
    }
    
    int Property::getTargetClassIndex()
    {
        return m_targetclassindex;
    }
    Property* Property::setTargetClassIndex(int targetclassindex)
    {
        m_targetclassindex = targetclassindex;
        return m_instance;
    }
    bool Property::getTargetMutil()
    {
        return m_targetmutil;
    }
    Property* Property::setTargetMutil(bool targetmutil)
    {
        m_targetmutil = targetmutil;
        return m_instance;
    }
    int Property::getTargetClasses()
    {
        return m_targetclasses;
    }
    Property* Property::setTargetClasses(int targetclasses)
    {
        m_targetclasses = targetclasses;
        return m_instance;
    }
    string Property::getRootPath()
    {
        return m_rootpath;
    }
    Property* Property::setRootPath(const string& rootpath)
    {
        m_rootpath = rootpath;
        return m_instance;
    }
    
    bool Property::getOtherDatasetHasHead()
    {
        return m_otherdatasethashead;
    }
    Property* Property::setOtherDatasetHasHead(bool otherdatasethashead)
    {
        m_otherdatasethashead = otherdatasethashead;
        return m_instance;
    }
    
    string Property::getLoggerPath()
    {
        return m_loggerpath;
    }
    Property* Property::setLoggerPath(const string& loggerpath)
    {
        m_loggerpath = loggerpath;
        return m_instance;
    }
    string Property::getTempPath()
    {
        return m_temppath;
    }
    Property* Property::setTempPath(const string& temppath)
    {
        m_temppath = temppath;
        return m_instance;
    }
    string Property::getClassName()
    {
        return m_classname;
    }
    Property* Property::setClassName(const string& classname)
    {
        m_classname = classname;
        return m_instance;
    }
    
    int Property::getNumTempDatasets()
    {
        return m_numtempdatasets;
    }
    Property* Property::setNumTempDatasets(int numtempdatasets)
    {
        m_numtempdatasets = numtempdatasets;
        return m_instance;
    }
    
    std::vector<string> Property::getDiscreteFeatureName()
    {
        return m_discretefeaturename;
    }
    Property* Property::setDiscreteFeatureName(const std::vector<string>& discretefeaturename)
    {
        m_discretefeaturename = discretefeaturename;
        return m_instance;
    }
    std::vector<string> Property::getDateFeatureName()
    {
        return m_datefeaturename;
    }
    Property* Property::setDateFeatureName(const std::vector<string> & datefeaturename)
    {
        m_datefeaturename = datefeaturename;
        return m_instance;
    }
    
    int Property::getDistributedNodes()
    {
        return m_distributednodes;
    }
    Property* Property::setDistributedNodes(int distributednodes)
    {
        m_distributednodes = distributednodes;
        return m_instance;
    }
    DataType Property::getDatasetType()
    {
		return m_datasettype;
	}
    Property* Property::setDatasetType(DataType datasettype)
    {
        m_datasettype = datasettype;
        return m_instance;
    }
    
    string Property::getTestDataPath()
    {
        return m_testdatapath;
    }
    Property* Property::setTestDataPath(const string& testdatapath)
    {
        m_testdatapath = testdatapath;
		return m_instance;
	}
    string Property::getDatasetPath()
    {
		return m_datasetpath;
	}
    Property* Property::setDatasetPath(const string& datasetpath)
    {
        m_datasetpath = datasetpath;
        return m_instance;
    }
    string Property::getSavePath()
    {
		return m_savepath;
	}
    Property* Property::setSavePath(const string& savepath)
    {
        m_savepath = savepath;
        return m_instance;
    }
    int Property::getMaxNumsFeatures()
    {
        return m_maxnumsfeatures;
    }
    Property* Property::setMaxNumsFeatures(int maxnumsfeatures)
    {
		m_maxnumsfeatures = maxnumsfeatures;
		return m_instance;
	}
    
    int Property::getFeatureNum()
    {
        return m_featurenum;
    }
    Property* Property::setFeatureNum(int featurenum)
    {
		m_featurenum = featurenum;
		return m_instance;
	}
    int Property::getLoggerLevel()
    {
        return m_loggerlevel;
    }
    Property* Property::setLoggerLevel(int loggerlevel)
    {
        m_loggerlevel = loggerlevel;
        return m_instance;
    }
    string Property::getMissVal()
    {
        return m_missval;
    }
    Property* Property::setMissVal(const string& missval)
    {
		m_missval = missval;
		return m_instance;
	}
    string Property::getValDataPath()
    {
        return m_valdatapath;
    }
    Property* Property::setValDataPath(const string& valdatapath)
    {
        m_valdatapath = valdatapath;
        return m_instance;
    }
    const std::unordered_map<string, string>& Property::getLightGBMParams()
    {
        return m_lightgbmparams;
    }
    Property* Property::setLightGBMParams(const std::unordered_map<string, string>& lightgbmparams)
    {
        m_lightgbmparams = lightgbmparams;
        return m_instance;
    }
    bool Property::getIsAllNumber()
    {
        return m_isallnumber;
    }
    Property* Property::setIsAllNumber(bool isallnumber)
    {
        m_isallnumber = isallnumber;
        return m_instance;
    }
    const std::unordered_map<string, string>& Property::getParams()
    {
        return m_params;
    }
    Property* Property::setParams(const std::unordered_map<string, string>& params)
    {
		m_params = params;
		return m_instance;
	}
    Property* Property::readProperty(string propertypath)
    {
        std::ifstream ifs(propertypath);
        if (!ifs.is_open())
        {
			LOG(ERROR) << "Can not open the property file!";
			return m_instance;
		}
        using json = nlohmann::json;
		json j = json::parse(ifs);
		ifs.close();
        if (j.find("classname") != j.end())
            m_instance->setClassName(j["classname"]);
        if (j.find("datasetname") != j.end())
            m_instance->setDatasetName(j["datasetname"]);
        if (j.find("datasettype") != j.end())
            m_instance->setDatasetType(j["datasettype"]);
        if (j.find("datefeaturename") != j.end())
            m_instance->setDateFeatureName(j["datefeaturename"]);
        if (j.find("distributednodes") != j.end())
            m_instance->setDistributedNodes(j["distributednodes"]);
        if (j.find("discretefeaturename") != j.end())
            m_instance->setDiscreteFeatureName(j["discretefeaturename"]);
        if (j.find("loggerpath") != j.end())
            m_instance->setLoggerPath(j["loggerpath"]);
        if (j.find("numtempdatasets") != j.end())
            m_instance->setNumTempDatasets(j["numtempdatasets"]);
        if (j.find("otherdatasethashead") != j.end())
            m_instance->setOtherDatasetHasHead(j["otherdatasethashead"]);
        if (j.find("rootpath") != j.end())
            m_instance->setRootPath(j["rootpath"]);
        if (j.find("targetclassindex") != j.end())
            m_instance->setTargetClassIndex(j["targetclassindex"]);
        if (j.find("targetclasses") != j.end())
            m_instance->setTargetClasses(j["targetclasses"]);
        if (j.find("targetmutil") != j.end())
            m_instance->setTargetMutil(j["targetmutil"]);
        if (j.find("temppath") != j.end())
            m_instance->setTempPath(j["temppath"]);
        if (j.find("threadnum") != j.end())
            m_instance->setThreadNum(j["threadnum"]);
        if (j.find("wethreadnum") != j.end())
            m_instance->setWeThreadNum(j["wethreadnum"]);
        if (j.find("testdatapath") != j.end())
            m_instance->setTestDataPath(j["testdatapath"]);
        if (j.find("datasetpath") != j.end())
            m_instance->setDatasetPath(j["datasetpath"]);
        if (j.find("savepath") != j.end())
            m_instance->setSavePath(j["savepath"]);
        if (j.find("maxnumsfeatures") != j.end())
            m_instance->setMaxNumsFeatures(j["maxnumsfeatures"]);
        if (j.find("loggerlevel") != j.end())
            m_instance->setLoggerLevel(j["loggerlevel"]);
        if (j.find("missval") != j.end())
            m_instance->setMissVal(j["missval"]);
        if (j.find("featurenum") != j.end())
            m_instance->setFeatureNum(j["featurenum"]);
        if (j.find("limitedmemory") != j.end())
			m_instance->setLimitedMemory(j["limitedmemory"]);
        if (j.find("valdatapath") != j.end())
            m_instance->setValDataPath(j["valdatapath"]);
        if (j.find("lightgbmparams") != j.end())
            m_instance->setLightGBMParams(j["lightgbmparams"]);
        if (j.find("isallnumber") != j.end())
            m_instance->setIsAllNumber(j["isallnumber"]);
        if (j.find("params") != j.end())
            m_instance->setParams(j["params"]);
        return m_instance;
    }
    Property::~Property() {  }
   
}//RL4FS


