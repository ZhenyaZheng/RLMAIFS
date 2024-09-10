#include "RL4FS/util/Tools.h"


namespace RL4FS {

    void GetFileNames(string path, std::vector<string>& filenames)
    {
        DIR* pDir;
        struct dirent* ptr;
        if (!(pDir = opendir(path.c_str()))) {
            // LOG(INFO) << path << " folder doesn't Exist!";
            return;
        }
        while ((ptr = readdir(pDir)) != 0) {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
                filenames.push_back(ptr->d_name);
            }
        }
        closedir(pDir);
        std::sort(filenames.begin(), filenames.end());
    }

    string getNowTime()
    {
        auto now = std::chrono::system_clock::now();
        
        uint64_t dis_millseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()
            - std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count() * 1000;
        time_t tt = std::chrono::system_clock::to_time_t(now);
        auto time_tm = localtime(&tt);
        char strTime[125] = { 0 };
        sprintf(strTime, "%d-%02d-%02d %02d:%02d:%02d %03d", time_tm->tm_year + 1900,
            time_tm->tm_mon + 1, time_tm->tm_mday, time_tm->tm_hour,
            time_tm->tm_min, time_tm->tm_sec, (int)dis_millseconds);
        return  string(strTime) ;
    }

    
    void clearFeatureData(DataSet* dataset, std::vector<PFeatureInfo>& featureinfos)
	{
		for (auto& featureinfo : featureinfos)
			dataset->clearFeatureData(featureinfo);
	}

  
    char* initMyMemory(const int n)
    {
        char* ptr = nullptr;
        ptr = new char[n + ADDMEMORY];
        if (!ptr)
            LOG(ERROR) << "initMyMemory Error: new char error!";
        return ptr;
    }

    bool clearMyMemory(char* &ptr)
    {
        delete[] ptr;
        ptr = nullptr;
        return true;
    }

    std::shared_timed_mutex* globalVar::getMutex()
    {
        if (!m_rwmutex)
            m_rwmutex = new  std::shared_timed_mutex;
        return  m_rwmutex;
    }
    bool globalVar::getIsReceive()
    {
        return m_isreceive;
    }
    void globalVar::setIsReceive(bool isreceive)
    {
        m_isreceive = isreceive;
    }
    int globalVar::getFCOperID()
    {
        return m_fcoperid;
    }
    void globalVar::setFCOperID(int fcoperid)
    {
        m_fcoperid = fcoperid;
    }

    std::shared_timed_mutex* globalVar::getMutex2()
    {
        if (!m_rwmutex2)
            m_rwmutex2 = new  std::shared_timed_mutex;
        return  m_rwmutex2;
    }
    bool globalVar::getIsReceive2()
    {
        return m_isreceive2;
    }
    void globalVar::setIsReceive2(bool isreceive)
    {
        m_isreceive2 = isreceive;
    }
    int globalVar::getFCOperID2()
    {
        return m_fcoperid2;
    }
    void globalVar::setFCOperID2(int fcoperid)
    {
        m_fcoperid2 = fcoperid;
    }

    globalVar* globalVar::getglobalVar()
    {
        if (m_instance == nullptr)
        {
            return m_instance = new globalVar();
        }
        else
            return m_instance;
    }
    bool globalVar::freeglobalVar()
    {
        delete m_instance;
        m_instance = nullptr;
        return true;
    }
}//RL4FS