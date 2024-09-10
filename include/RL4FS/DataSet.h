#pragma once
#include "./FeatureInfo.h"
#include "./util/StopWatch.h"
#include "thundersvm/dataset.h"
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <shared_mutex>

namespace RL4FS {
#define WAITTIMEDIVSECOND 10000000

	class DataSet
	{
	public:
		using node = ThunderSVM::DataSet::node;
		using node2d = ThunderSVM::DataSet::node2d;
		DataSet();
		DataSet(const DataSet& data);
		~DataSet();
		DataSet& operator= (const DataSet & data);
		DataSet(std::vector<PFeatureInfo> features, PFeatureInfo target, string name, int instancesoffeature, int id=0, int numid =1);
		bool addFeature(PFeatureInfo &&);
		bool relloc();
		void setFeatures(const std::vector<PFeatureInfo> & featureinfos);
		void setFeature(int i, const PFeatureInfo& featureinfo);
		void setTargetFeature(const PFeatureInfo & featureinfo);
		void setInstancesOfFeature(int instancesoffeature);
		bool addFeature(const PFeatureInfo& );
		PFeatureInfo popFeature(bool onlydata=false);
		std::vector<PFeatureInfo> getFeatures();
		std::pair<std::vector<PFeatureInfo>, PFeatureInfo> getFeatures(bool);
		PFeatureInfo getTargetFeature();
		PFeatureInfo getFeature(int index, bool getdata=false);
		DataSet* deepcopy();
		DataSet* copy(char * ptr);
		bool write(const string & path, const string & other="");
		int getInstancesOfFeature(int type=1);//
		int getFeatureSize(bool = false);//
		bool serialize(int lineindex = -1);
		bool deserialize(int lineindex = -1);
		int getID() { return m_id; }
		void setID(int id) { m_id = id; }
		int getNumID() { return m_numid; }
		std::vector<int> getDatasetIndex() { return m_datasetindex; }
		void setDatasetIndex(std::vector<int>& datasetindex) { m_datasetindex = datasetindex; }
		string getName();//
		std::shared_timed_mutex* getMutex();
		PFeatureInfo getFeatureFromOld(const PFeatureInfo& featureinfo, bool needdata = true);
		void clear();
		void UpdateNameFeature();
		void setNumOfDatasetInstances(const std::vector<int>&);
		std::vector<int> getNumOfDatasetInstances() { return m_numofdatasetinstances; }
		int getDistributedInstancesofFeature() { return m_distributedinstancesoffeature; }
		void getInstancesTfromThunder();
		PFeatureInfo getFeatureFromInstanceT(const PFeatureInfo& featureinfo);
		PFeatureInfo getFeatureFromInstanceT(int featureindex);
		void LoadThunderData(string datapath, int classes = Property::getProperty()->getTargetClasses(), string datasetname = Property::getProperty()->getDatasetName());
		void LoadThunderDatas(string datapath, std::vector<string>datanames, int classes = Property::getProperty()->getTargetClasses(), string datasetname = Property::getProperty()->getDatasetName());
		void Init();
		void clearFeatureData(int featureindex);
		void clearFeatureData(const PFeatureInfo& featureinfo);
		int getFeatureIndex(const PFeatureInfo& featureinfo);
		ThunderSVM::DataSet* getThunderData() { return m_thunderdata; }
		void setFeatureFromOld(const PFeatureInfo& oldfeatureinfo, PFeatureInfo &featureinfo);
		int getDiscreteFeatureSize(){return m_discretefeaturesize;}
		void getT(node2d *sources, node2d* instances, int newm, int oldm, bool iszerobased);
		void setFeatureSize(int featuresize) { m_featuresize = featuresize; }
		void setDistributedInstancesofFeature(int distributedinstancesoffeature) { m_distributedinstancesoffeature = distributedinstancesoffeature; }
		//void setNameFeatureIndex(const std::unordered_map<string, int>& namefeatureindex){m_namefeatureindex = namefeatureindex;}
		void setInstancesT(const node2d& instancesT) { m_instancesT = instancesT; }
		void setDiscreteFeatureSize(int discretefeaturesize) { m_discretefeaturesize = discretefeaturesize; }
		void setFeatureLock(const std::vector<std::shared_timed_mutex*>& featurelock);
		std::vector<std::shared_timed_mutex*> getFeatureLock();
		std::vector<int> getFeatureUsed() { return m_featureused; }
		void setFeatureUsed(const std::vector<int>& featureused) { m_featureused = featureused; }
		int getMaxFeatureUsed() { return m_maxfeatureused; }
		void setMaxFeatureUsed(int maxfeatureused) { m_maxfeatureused = maxfeatureused; }
		int getFeatureUsedNum() { return m_featureusednum; }
		void setFeatureUsedNum(int featureusednum){m_featureusednum = featureusednum;}
		void setSparseInstancesPerFeature(const std::vector<int>& sparseinstancesperfeature) { m_sparseinstancesperfeature = sparseinstancesperfeature; }
		std::vector<int> getSparseInstancesPerFeature() { return m_sparseinstancesperfeature; }
		unsigned long long getMemorySize();
		std::vector<long long> getNumPoints() { return m_numpoints; }
		void setNumPoints(const std::vector<long long>& numpoints) { m_numpoints = numpoints; }
		void updateNumPoints();
		int megerDataSet(DataSet* dataset, int datatype=0);
		void setFeatureInstanceT(const PFeatureInfo& featureinfo);
		void setFeatureInstanceT(int featureindex);
		void setSelectFeatures(const std::vector<int>& selectedfeatures) { m_selectedfeatures = selectedfeatures; }
		const std::vector<int>& getSelectFeatures() { return m_selectedfeatures; }
		void updateDataSet();
	private:	
		std::vector<PFeatureInfo> m_features;// the features of dataset
		PFeatureInfo m_targetfeature;// the target feature of dataset
		int m_featuresize;// the size of features
		int m_instancesoffeature;// the number of instances of feature
		string m_name;// the name of dataset
		volatile std::atomic_int m_id;// the id of dataset
		int m_numid;// the number of dataset chunks
		std::vector<int> m_datasetindex;// the index of dataset chunks
		std::vector<int> m_numofdatasetinstances;// the number of instances of dataset chunks
		std::shared_timed_mutex* m_rwmutex = nullptr;// the mutex of dataset
		int m_distributedinstancesoffeature;// the number of distributed instances of feature
		ThunderSVM::DataSet* m_thunderdata = nullptr;// the data of sparse dataset
		// std::unordered_map<string, int> m_namefeatureindex;// the index of feature name
		node2d m_instancesT;// the instances of feature for sparse dataset
		int m_discretefeaturesize = 0;// the size of discrete features
		std::vector<std::shared_timed_mutex*> m_featurelock;// the lock of feature
		std::vector<int> m_featureused;// the used features
		int m_maxfeatureused = 0;// the maximum used feature
		volatile std::atomic_int m_featureusednum = 0;// the number of used features
		std::vector<int> m_sparseinstancesperfeature;// the number of sparse instances of all features
		std::vector<long long> m_numpoints;// the number of points
		std::vector<int> m_selectedfeatures;// the selected features
	public:
		std::vector<int> m_trainindex;
		std::vector<int> m_valindex;
		std::vector<int> m_testindex;
		bool m_hasval = false;
		bool m_hastest = false;
	};

	class TempDataSet
	{
		std::vector<DataSet* > m_tempdataset;
		volatile std::atomic_int m_index;
		std::shared_timed_mutex* volatile m_rwmutex = nullptr;
	public:
		TempDataSet();
		TempDataSet(const std::vector<DataSet* >& tempdataset);
		~TempDataSet();
		DataSet* getNext();
		int getIndex();
		std::vector<DataSet*> getDataSets();
		void Init(const std::vector<DataSet* >& tempdataset);
		void Release();
		DataSet* getDataSet();
		std::shared_timed_mutex* getMutex();
	};

}//RL4FS



