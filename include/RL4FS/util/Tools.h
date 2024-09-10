#pragma once
#include "StopWatch.h"
#include "DataSet.h"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <omp.h>
#include <shared_mutex>
//#include "./MemoryLeak.h"

namespace RL4FS {

#define ADDMEMORY 100
	char* initMyMemory(const int n);
	bool clearMyMemory(char*& ptr);
    string getNowTime();
    template<typename T>
    void RL4FSClear(T& vecs)
    {
        for (int i = 0; i < vecs.size(); ++i)
        {
            try {
                if (vecs[i])
                {
                    vecs[i]->clear();
                    delete vecs[i];
                    vecs[i] = nullptr;
                }
            }
            catch (exception& e)
            {
                LOG(WARNING) << "RL4FSClear error : " << e.what();
                continue;
            }
        }
        vecs.clear();
    }
	void GetFileNames(string path, std::vector<string>& filenames);
    void clearFeatureData(DataSet* dataset, std::vector<PFeatureInfo>& featureinfos);
    
    class ClearClass;
    class globalVar
    {
        class FCOperID
        {
        public:
            string m_firstname;
            string m_secondname;
            std::vector<string> m_sourcename;
            std::vector<string> m_targetname;
        
            bool operator==(const FCOperID& fcoperid)
            {
                if (m_firstname == fcoperid.m_firstname && m_secondname == fcoperid.m_secondname && m_sourcename == fcoperid.m_sourcename && m_targetname == fcoperid.m_targetname)
                    return true;
                return false;
            }
            bool operator!=(const FCOperID& fcoperid) { return !(*this == fcoperid); }
            FCOperID() {}
            FCOperID(const string& name1, const string& name2, const std::vector<string>& name3, const std::vector<string>& name4) :m_firstname(name1), m_secondname(name2), m_sourcename(name3), m_targetname(name4) {}
        };
    public:
        friend ClearClass;
        std::shared_timed_mutex* getMutex();
        bool getIsReceive();
        void setIsReceive(bool isreceive);
        int getFCOperID();
        void setFCOperID(int fcoperid);
        std::shared_timed_mutex* getMutex2();
        bool getIsReceive2();
        void setIsReceive2(bool isreceive);
        int getFCOperID2();
        void setFCOperID2(int fcoperid);
        static globalVar* getglobalVar();
        static bool freeglobalVar();
        ~globalVar() { delete m_rwmutex; delete m_rwmutex2; m_rwmutex = nullptr; m_rwmutex2 = nullptr; }
    private:
        static globalVar* m_instance;
        std::shared_timed_mutex* volatile m_rwmutex = nullptr;
        int m_fcoperid = 0;
        bool m_isreceive = false;
        std::shared_timed_mutex* volatile m_rwmutex2 = nullptr;
        int m_fcoperid2 = 0;
        bool m_isreceive2 = false;
    };

}//RL4FS