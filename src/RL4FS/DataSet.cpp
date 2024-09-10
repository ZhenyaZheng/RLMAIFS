#include "RL4FS/DataSet.h"
namespace RL4FS {
	DataSet::DataSet() :m_featuresize(0), m_instancesoffeature(0), m_targetfeature(nullptr), m_id(0), m_numid(1), m_rwmutex(nullptr) { getMutex(); }
	
	DataSet::DataSet(const DataSet& data)
	{
		m_features = data.m_features;
		m_targetfeature = data.m_targetfeature;
		m_name = data.m_name;
		m_instancesoffeature = data.m_instancesoffeature;
		m_featuresize = data.m_featuresize;
		m_id = data.m_id;
		m_numid = data.m_numid;
		m_datasetindex = data.m_datasetindex;
		m_numofdatasetinstances = data.m_numofdatasetinstances;
		m_rwmutex = data.m_rwmutex;
		m_distributedinstancesoffeature = data.m_distributedinstancesoffeature;
		m_thunderdata = data.m_thunderdata;
		m_instancesT = data.m_instancesT;
		// m_namefeatureindex = data.m_namefeatureindex;
	}
	
	DataSet& DataSet::operator= (const DataSet& data)
	{
		m_features = data.m_features;
		m_targetfeature = data.m_targetfeature;
		m_name = data.m_name;
		m_instancesoffeature = data.m_instancesoffeature;
		m_featuresize = data.m_featuresize;
		m_id = data.m_id;
		m_numid = data.m_numid;
		m_datasetindex = data.m_datasetindex;
		m_numofdatasetinstances = data.m_numofdatasetinstances;
		m_rwmutex = data.m_rwmutex;
		m_distributedinstancesoffeature = data.m_distributedinstancesoffeature;
		m_thunderdata = data.m_thunderdata;
		m_instancesT = data.m_instancesT;
		// m_namefeatureindex = data.m_namefeatureindex;
		return *this;
	}
	
	void DataSet::clear()
	{
		m_features.clear();
		m_features.~vector();
		m_name.clear();
		m_name.~basic_string();
		m_datasetindex.clear();
		m_datasetindex.~vector();
		m_numofdatasetinstances.clear();
		m_numofdatasetinstances.~vector();
	}
	
	DataSet::DataSet(std::vector<PFeatureInfo> features, PFeatureInfo target, string name, int instancesoffeature, int id, int numid): m_rwmutex(nullptr)
	{
		m_features = features;
		m_targetfeature = target;
		m_name = name;
		m_instancesoffeature = instancesoffeature;
		m_featuresize = features.size() + 1;
		m_id = id;
		m_numid = numid;
		Init();
	}
	
	void DataSet::Init()
	{
		m_datasetindex.resize(static_cast<size_t>(m_numid + 1));
		UpdateNameFeature();
		getMutex();
		m_featureused.resize(m_featuresize - 1, 0);
		m_featurelock.resize(m_featuresize - 1, nullptr);
		m_maxfeatureused = (10ULL << 30) / (sizeof(MyDataType) * m_instancesoffeature);
	}
	
	DataSet::~DataSet()
	{
		for (int i = 0; i < m_features.size(); ++i)
			m_features[i]->clear(), delete m_features[i], m_features[i] = nullptr;
		m_features.clear();
		if (m_targetfeature)
		{
			m_targetfeature->clear();
			delete m_targetfeature;
			m_targetfeature = nullptr;
		}

		if (m_rwmutex)
		{
			delete m_rwmutex;
			m_rwmutex = nullptr;
		}
		m_datasetindex.clear();
		m_numofdatasetinstances.clear();
		
		if (m_thunderdata)
		{
			delete m_thunderdata;
			m_thunderdata = nullptr;
		}
		m_instancesT.clear();
		//m_namefeatureindex.clear();
		m_featureused.clear();
		for (auto& featurelock : m_featurelock)
		{
			if (featurelock)
			{
				delete featurelock;
				featurelock = nullptr;
			}
		}
		m_featurelock.clear();
	}
	
	DataSet* DataSet::deepcopy()
	{
		std::vector<PFeatureInfo> featureinfos;
		std::unordered_map<string, PFeatureInfo> pfmap;
		std::vector<PFeatureInfo> temppf;
		for (auto& featureinfo : m_features)
		{
			PFeatureInfo newfeatureinfo = new FeatureInfo();
			newfeatureinfo->copy(featureinfo);
			pfmap[newfeatureinfo->getName()] = newfeatureinfo;
			
			if (newfeatureinfo->getIsSecondFeature())
			{
				auto newfsf = newfeatureinfo->getSourceFeatures()[0];
				temppf.clear();
				for (auto& sfeinfo : newfsf->getSourceFeatures())
				{
					if (pfmap[sfeinfo->getName()])
						temppf.push_back(pfmap[sfeinfo->getName()]);
					else LOG(ERROR) << "DataSet deepcopy Error!";
				}
				if (temppf.size())
					newfsf->setSourceFeatures(temppf);
				temppf.clear();
				for (auto& sfeinfo : newfsf->getTargetFeatures())
				{
					if (pfmap[sfeinfo->getName()])
						temppf.push_back(pfmap[sfeinfo->getName()]);
					else LOG(ERROR) << "DataSet deepcopy Error!";
				}
				if (temppf.size())
					newfsf->setTargetFeatures(temppf);
				featureinfos.push_back(newfeatureinfo);
				continue;
			}
			temppf.clear();
			for (auto& sfeinfo : newfeatureinfo->getSourceFeatures())
			{
				if (pfmap[sfeinfo->getName()])
					temppf.push_back(pfmap[sfeinfo->getName()]);
				else LOG(ERROR) << "DataSet deepcopy Error!";
			}
			if(temppf.size())
				newfeatureinfo->setSourceFeatures(temppf);
			temppf.clear();
			for (auto& sfeinfo : newfeatureinfo->getTargetFeatures())
			{
				if (pfmap[sfeinfo->getName()])
					temppf.push_back(pfmap[sfeinfo->getName()]);
				else LOG(ERROR) << "DataSet deepcopy Error!";
			}
			if (temppf.size()) 
				newfeatureinfo->setTargetFeatures(temppf);
			featureinfos.push_back(newfeatureinfo);
		}
		PFeatureInfo targetinfo = new FeatureInfo();
		targetinfo->copy(m_targetfeature);
		auto newdataset = new DataSet(featureinfos, targetinfo, m_name, m_instancesoffeature, m_id, m_numid);
		newdataset->setDatasetIndex(m_datasetindex);
		newdataset->setNumOfDatasetInstances(m_numofdatasetinstances);
		
		newdataset->setFeatureSize(m_featuresize);
		newdataset->setDistributedInstancesofFeature(m_distributedinstancesoffeature);
		//newdataset->setNameFeatureIndex(m_namefeatureindex);
		newdataset->setInstancesT(m_instancesT);
		newdataset->setDiscreteFeatureSize(m_discretefeaturesize);
		newdataset->setFeatureLock(m_featurelock);
		newdataset->setFeatureUsed(m_featureused);
		newdataset->setMaxFeatureUsed(m_maxfeatureused);
		newdataset->setFeatureUsedNum(m_featureusednum);
		newdataset->setSparseInstancesPerFeature(m_sparseinstancesperfeature);
		newdataset->setNumPoints(m_numpoints);
		newdataset->m_trainindex = m_trainindex;
		newdataset->m_valindex = m_valindex;
		newdataset->m_testindex = m_testindex;
		newdataset->m_hasval = m_hasval;
		newdataset->m_hastest = m_hastest;
		return newdataset;
	}

	void DataSet::setNumOfDatasetInstances(const std::vector<int> & numofdatasetinstances)
	{
		m_numofdatasetinstances = numofdatasetinstances;
	}

	DataSet* DataSet::copy(char* ptr)
	{
		memset(ptr, 0, sizeof (DataSet));
		ptr = reinterpret_cast<char*>( new(ptr) DataSet(*this));
		return reinterpret_cast<DataSet*> (ptr);
	}
	
	void DataSet::getT(node2d *sources, node2d *instances, int newm, int oldm, bool iszerobased)
	{
		if (oldm != sources->size())
		{
			LOG(ERROR) << "DataSet getT Error: oldm != sources->size()!";
			exit(-1);
		}
		instances->resize(newm);
		// multi thread
		int threadnums = Property::getProperty()->getThreadNum();
		for (int i = 0; i < oldm; ++i)
		{
			auto instance = sources->data() + i;
			int realn = instance->size();
			if (realn < threadnums) threadnums = realn / 100;
			if (threadnums < 1 || realn <= 10) threadnums = 1;
			int step = realn / threadnums;
			const int ompthreadnums = threadnums;
#pragma omp parallel num_threads(ompthreadnums)
			{
				int tid = omp_get_thread_num();
				int start = tid * step;
				int end = (tid + 1) * step;
				if (tid == threadnums - 1) end = realn;
				for (int j = start; j < end; ++j)
				{
					auto pnode = instance->data() + j;
					auto index = iszerobased ? pnode->index : pnode->index - 1;
					instances->at(index).emplace_back(node(i, pnode->value));
				}
			}
			instance->clear();
			instance->shrink_to_fit();
		}
		sources->clear();
		sources->shrink_to_fit();
		LOG(INFO) << "DataSet getT successful!";
	}

	void DataSet::setFeatureLock(const std::vector<std::shared_timed_mutex*>& featurelock)
	{
		m_featurelock = featurelock;
	}
	
	std::vector<std::shared_timed_mutex*> DataSet::getFeatureLock()
	{
			return m_featurelock;
	}

	unsigned long long DataSet::getMemorySize()
	{
		unsigned long long msize = 0;
		if(m_instancesT.size())
		{
			msize += m_instancesoffeature * sizeof(MyDataType);
			auto featruesize = m_instancesT.size();
			for (int i = 0; i < featruesize; ++i)
				msize += m_instancesT[i].size() * sizeof(node);
		}
		else
			msize = m_featuresize * m_instancesoffeature * sizeof(MyDataType);
		return msize;
	}

	void DataSet::updateNumPoints()
	{
		m_numpoints.resize(m_numid, 0);
		m_numpoints[m_id] = 0;
		if (m_instancesT.size())
		{
			if (m_numid > 1)
			{
				for (int i = 0; i < m_sparseinstancesperfeature.size(); ++i)
					m_numpoints[m_id] += m_sparseinstancesperfeature[i];
			}
			else
			{
				for (int i = 0; i < m_instancesT.size(); ++i)
					m_numpoints[m_id] += m_instancesT[i].size();
			}
		}
		else
			m_numpoints[m_id] = (m_featuresize -1) * m_instancesoffeature;
	}

	
	
	bool DataSet::write(const string& savepath, const string& other)
	{
		try {
			string filepath = savepath;
			if (other == "")
				filepath += m_name;
			else filepath += other;
			if (m_numid > 1)
			{
#if defined(_WIN32) || defined(_WIN64)
				auto createflag = _mkdir(filepath.c_str());
#else
				auto createflag = mkdir(filepath.c_str(), 0777);
#endif
				if (createflag >= 0)
					LOG(INFO) << filepath << " create successful!";
			}
			else 
			{
				// if savepath is not exist, create it
				if(!std::filesystem::exists(savepath))
            	{
					std::filesystem::create_directories(savepath);
					LOG(INFO) << savepath << " create successful!";
				}
			}
			string endformat = ".csv";
			if (m_thunderdata || m_instancesT.size() > 0)
				endformat = "";
			for (int id = 0; id < m_numid; ++id)
			{
				string path = filepath;
				string valpath = path + "_val";
				string testpath = path + "_test";
				if (m_numid > 1)
					path += "/" + m_name + std::to_string(id) + "_afterfs" + endformat;
				else if (other == "")
					path += "_afterfs" + endformat, valpath += "_afterfs" + endformat, testpath += "_afterfs" + endformat;
				else path += endformat, valpath += endformat, testpath += endformat;
				if(!m_hasval) valpath = "";
				if(!m_hastest) testpath = "";
				setID(id);
				deserialize();
				ofstream file(path);
				ofstream valfile(valpath);
				ofstream testfile(testpath);
				if (!file.is_open())
				{
					LOG(ERROR) << "DataSet write Error: " << path << " open failed, Please create directory firstly!";
					return false;
				}
				int targetindex = m_datasetindex[id];
				if (m_thunderdata || m_instancesT.size() > 0)
				{
					node2d instances;
					getT(&m_instancesT, &instances, m_instancesoffeature, m_featuresize - 1, true);
					auto* pthunderdata = &instances;
					for (int i = 0; i < (*pthunderdata).size(); ++i)
					{
						ofstream* pfile = &file;
						if(m_hasval && i >= m_valindex[0]) pfile = &valfile;
						if(m_hastest && i >= m_testindex[0]) pfile = &testfile;
						if (m_targetfeature->getType() == OutType::Discrete)
							*pfile << *reinterpret_cast<int*>(m_targetfeature->getFeature()->getValue(i + targetindex));
						else *pfile << *reinterpret_cast<MyDataType*>(m_targetfeature->getFeature()->getValue(i + targetindex));
						for (int j = 0; j < (*pthunderdata)[i].size(); ++j)
							*pfile << ' ' << (*pthunderdata)[i][j].index << ':' << (*pthunderdata)[i][j].value;
						*pfile << '\n';
					}
				}
				else
				{
					for (auto& feature : m_features)
					{
						file << feature->getName() << ',';
						if(m_hasval) valfile << feature->getName() << ',';
						if(m_hastest) testfile << feature->getName() << ',';
					}
					file << m_targetfeature->getName() << "\n";
					if(m_hasval) valfile << m_targetfeature->getName() << "\n";
					if(m_hastest) testfile << m_targetfeature->getName() << "\n";
					for (int i = 0; i < m_instancesoffeature; ++i)
					{
						ofstream* pfile = &file;
						if (m_hasval && i >= m_valindex[0]) pfile = &valfile;
						if (m_hastest && i >= m_testindex[0]) pfile = &testfile;
						for (auto& feature : m_features)
						{
							auto type = feature->getFeature()->getType();
							switch (type)
							{
							case FeatureType::Date:
								*pfile << *reinterpret_cast<Date*>(feature->getFeature()->getValue(i));
								break;
							case FeatureType::String:
								*pfile << *reinterpret_cast<string*>(feature->getFeature()->getValue(i));
								break;
							case FeatureType::Numeric:
								*pfile << *reinterpret_cast<MyDataType*>(feature->getFeature()->getValue(i));
								break;
							case FeatureType::Discrete:
								*pfile << *reinterpret_cast<int*>(feature->getFeature()->getValue(i));
								break;
							}
							*pfile << ',';
						}
						auto classes = Property::getProperty()->getTargetClasses();
						if (classes)
							*pfile << *reinterpret_cast<int*>(m_targetfeature->getFeature()->getValue(targetindex + i)) << "\n";
						else *pfile << *reinterpret_cast<MyDataType*>(m_targetfeature->getFeature()->getValue(targetindex + i)) << "\n";
					}
				}
				file.close();
			}
		}
		catch (exception& e)
		{
			LOG(ERROR) << "DataSet write Error: " << e.what();
			return false;
		}
		LOG(INFO) << "DataSet " << m_name << " write successful!";
		return true;
	}
	
	bool DataSet::addFeature(const PFeatureInfo & fi)
	{
		m_features.push_back(fi);
		//m_namefeatureindex[fi->getName()] = m_featuresize - 1;
		fi->setIndex(m_featuresize - 1);
		m_featuresize++;
		m_featureused.push_back(0);
		m_featurelock.push_back(nullptr);
		if (m_thunderdata || m_instancesT.size())
		{
			if (m_numid == 1)m_rwmutex->lock();
			if (!m_featurelock[m_featuresize - 2])m_featurelock[m_featuresize - 2] = new std::shared_timed_mutex;
			m_featureused[m_featuresize - 2] = 1;
			if (m_numid == 1)m_rwmutex->unlock();
			auto feature = fi->getFeature();
			if (!feature)
			{
				LOG(ERROR) << "DataSet addFeature Error: feature is nullptr!";
				exit(-1);
			}
			std::vector<node> tempcol;
			// auto instances = const_cast<node2d*>(& m_thunderdata->instances());
			for (int i = 0; i < m_instancesoffeature; ++i)
			{
				if (feature->getType() == FeatureType::Discrete)
				{
					auto value = *reinterpret_cast<int*> (feature->getValue(i));
					if (value != 0)
					{
						tempcol.push_back(node(i, static_cast<float_type>(value)));
						// (*instances)[i].push_back(node(m_featuresize - 1, static_cast<float_type>(value)));
					}
				}
				else if (feature->getType() == FeatureType::Numeric)
				{
					auto value = *reinterpret_cast<MyDataType*> (feature->getValue(i));
					if (value != 0)
					{
						tempcol.push_back(node(i, static_cast<float_type>(value)));
					}
				}
			}
			
			m_instancesT.push_back(tempcol);
			clearFeatureData(fi);
			m_sparseinstancesperfeature.push_back(static_cast<int>(tempcol.size()));
			m_numpoints[m_id] += static_cast<long long>(tempcol.size());
		}
		else m_numpoints[m_id] += m_instancesoffeature;
		if (fi->getType() == OutType::Discrete)
				m_discretefeaturesize++;
		return true;
	}
	
	bool DataSet::relloc()
	{
		auto n = m_features.size();
		if(m_features.capacity() == n)
			m_features.reserve(n + 10);
		return true;
	}

	bool DataSet::addFeature(PFeatureInfo && fi)
	{
		addFeature(fi);
		return true;
	}
	
	PFeatureInfo DataSet::popFeature(bool onlydata)
	{
		int fsize = m_features.size();
		if (fsize <= 0)
		{
			LOG(ERROR) << "RL4FS::DataSet popFeature error: m_features is empty";
			return nullptr;
		}
		auto fi = m_features[fsize - 1]; 
		if (fi->getType() == OutType::Discrete)
			m_discretefeaturesize--;
		m_features.pop_back(); 
		m_featuresize--; 
		m_featureused.pop_back();
		if (m_thunderdata || m_instancesT.size())
		{
			if(!onlydata) m_numpoints[m_id] -= m_sparseinstancesperfeature[m_featuresize - 1];
			if (m_featurelock[m_featuresize - 1])
			{
				delete m_featurelock[m_featuresize - 1];
				m_featurelock[m_featuresize - 1] = nullptr;
			}
			m_instancesT.pop_back();
			m_sparseinstancesperfeature.pop_back();
		}
		else if(!onlydata) m_numpoints[m_id] -= m_instancesoffeature;
		m_featurelock.pop_back();

		return fi; 
	}
	
	void DataSet::setFeatures(const std::vector<PFeatureInfo>& featureinfos)
	{
		m_features = featureinfos;
	}
	
	void DataSet::setFeature(int i, const PFeatureInfo& featureinfo)
	{
		m_features[i] = featureinfo;
	}
	
	void DataSet::setTargetFeature(const PFeatureInfo& featureinfo)
	{
		m_targetfeature = featureinfo;
	}
	
	std::vector<PFeatureInfo> DataSet::getFeatures()
	{
		return this->m_features;
	}
	
	std::pair<std::vector<PFeatureInfo>, PFeatureInfo> DataSet::getFeatures(bool istrain)
	{
		return make_pair(this->m_features, this->m_targetfeature);
	}
	
	PFeatureInfo DataSet::getTargetFeature()
	{
		return this->m_targetfeature;
	}
	
	PFeatureInfo DataSet::getFeature(int index, bool getdata)
	{
		if (m_instancesT.size() && getdata)
		{
			if(m_featurelock[index] == nullptr)m_featurelock[index] = new std::shared_timed_mutex;
			while (m_featureusednum >= m_maxfeatureused)
			{
				LOG(DEBUG) << "DataSet getFeature: m_featureusednum = " << m_featureusednum << " m_maxfeatureused = " << m_maxfeatureused;
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
			m_featurelock[index]->lock();
			m_featureused[index]++;
			if (m_featureused[index] == 1)
			{
				getFeatureFromInstanceT(index);
				m_featureusednum++;
			}
			m_featurelock[index]->unlock();
		}
		return (this->m_features)[index];
	}
	
	int DataSet::getFeatureSize(bool istest)
	{
		if (istest)
		{
			return m_featuresize;
		}
		else
		{
			return m_featuresize - 1;
		}
	}

	int DataSet::getInstancesOfFeature(int type)
	{
		if (type == 2)
		{
			if (Property::getProperty()->getDistributedNodes() > 1)
				return this->m_distributedinstancesoffeature;
		}
		else if (type == 3)
		{
			if (m_numid > 1)
			{
				/*int sum = 0;
				for (int i = 0; i < m_numid; ++i)
					sum += m_numofdatasetinstances[i];
				return sum;*/
				return m_datasetindex[m_numid];
			}
		}
		return this->m_instancesoffeature;
	}
	
	void DataSet::setInstancesOfFeature(int instancesoffeature)
	{
		this->m_instancesoffeature = instancesoffeature;
	}
	
	string DataSet::getName()
	{
		return m_name;
	}
	
	bool DataSet::serialize(int lineindex)
	{
		if (m_numid == 1)return true;
		if (m_numofdatasetinstances.size() == 0)
			m_numofdatasetinstances.resize(m_numid);
		m_numofdatasetinstances[m_id] = m_instancesoffeature;
		
		
		string filedirpath = Property::getProperty()->getTempPath();
#if defined(_WIN32) || defined(_WIN64)
		auto createflag = _mkdir(filedirpath.c_str());
#else
		auto createflag = mkdir(filedirpath.c_str(), 0777);
#endif
		if (createflag >= 0)
			LOG(INFO) << filedirpath << " create successful!";
		
		auto datafilepath = filedirpath + getName() + std::to_string(m_id) + ".data";
		
		std::ofstream datafile;
		// judge the file is exist or not
		// if not exist, create it
		// if exist, open it
		datafile.open(datafilepath, ios::in | ios::ate | ios::binary );
		if (! datafile.good())
		{
			datafile.close();
			datafile.open(datafilepath, ios::out | ios::binary);
		}
		auto& file = datafile;
		std::unordered_map<FeatureType, int> sizeoftype{ {FeatureType::Date, sizeof (Date)},{FeatureType::String, sizeof (string)},
			{FeatureType::Numeric, sizeof(MyDataType)},{FeatureType::Discrete, sizeof (int)} };
		
		updateNumPoints();
		if (file.is_open())
		{
			int n = m_instancesoffeature;
			unsigned long long offset = 0;
			file.seekp(offset, ios::beg);
			if (lineindex >= 0)
			{
				if (lineindex >= m_featuresize - 1)
				{
					LOG(ERROR) << "DataSet serialize Error: lineindex >= m_featuresize - 1!";
					file.close();
					return false;
				}
				for (int i = 0; i <= lineindex; ++i)
				{	
					if (i == lineindex)
					{
						file.seekp(offset, ios::beg);
						if (m_thunderdata || m_instancesT.size())
							file.write(reinterpret_cast<char*>(m_instancesT[i].data()), static_cast<size_t>(m_sparseinstancesperfeature[i]) * sizeof(node));
						else {
							auto type = m_features[i]->getFeature()->getType();
							file.write(reinterpret_cast<char*> (m_features[i]->getFeature()->getValues()), static_cast<size_t>(n) * sizeoftype[type]);
						}
						break;
					}
					if (m_thunderdata || m_instancesT.size())
						offset += sizeof(node) * m_sparseinstancesperfeature[i];
					else
					{
						auto type = m_features[i]->getFeature()->getType();
						offset += sizeoftype[type];
					}

				}
			}
			else
			{
				for (int i = 0; i < m_featuresize - 1; ++i)
				{
					if (m_thunderdata || m_instancesT.size())
						file.write(reinterpret_cast<char*>(m_instancesT[i].data()), static_cast<size_t>(m_sparseinstancesperfeature[i]) * sizeof(node));
					else {
						auto type = m_features[i]->getFeature()->getType();
						file.write(reinterpret_cast<char*> (m_features[i]->getFeature()->getValues()), static_cast<size_t>(n) * sizeoftype[type]);
					}
				}
			}
			if (m_thunderdata || m_instancesT.size())
			{
				offset = m_numpoints[m_id] * sizeof(node);
				file.seekp(offset, ios::beg);
				file.write(reinterpret_cast<char*>(m_sparseinstancesperfeature.data()), static_cast<size_t>(m_featuresize - 1) * sizeof(int));
			}
		}
		else
		{
			LOG(ERROR) << datafilepath << " create failed!";
			file.close();
			return false;
		}
		file.close();
		return true;
	}
	
	bool DataSet::deserialize(int lineindex)
	{
		if (m_numid == 1)return true;
		string filedirpath = Property::getProperty()->getTempPath();
		
		auto datafilepath = filedirpath + getName() + std::to_string(m_id) + ".data";
		std::ifstream datafile;
		datafile.open(datafilepath, ios::in | ios::binary);
		auto& file = datafile;
		std::unordered_map<FeatureType, int> sizeoftype{ {FeatureType::Date, sizeof(Date)}
		,{FeatureType::String, sizeof(string)},{FeatureType::Numeric, sizeof(MyDataType)},{FeatureType::Discrete, sizeof(int)} };
		if (file.is_open())
		{
			int n = m_numofdatasetinstances[m_id];
			setInstancesOfFeature(n);
			long long offset = 0;
			if (Property::getProperty()->getDatasetType() == DataType::LibSVMCF || Property::getProperty()->getDatasetType() == DataType::LibSVMRG)
			{
				offset = m_numpoints[m_id] * sizeof(node);
				file.seekg(offset, ios::beg);
				file.read(reinterpret_cast<char*>(m_sparseinstancesperfeature.data()), static_cast<size_t>(m_featuresize - 1) * sizeof(int));
				m_instancesT.resize(m_featuresize - 1);
			}
			offset = 0;
			if (lineindex >= 0)
			{
				if (lineindex >= m_featuresize - 1)
				{
					LOG(ERROR) << "DataSet deserialize Error: lineindex >= m_featuresize - 1!";
					file.close();
					return false;
				}
				for (int i = 0; i <= lineindex; ++i)
				{
					if (i == lineindex)
					{
						file.seekg(offset, ios::beg);
						if (Property::getProperty()->getDatasetType() == DataType::LibSVMCF || Property::getProperty()->getDatasetType() == DataType::LibSVMRG)
						{
							m_instancesT[i].resize(m_sparseinstancesperfeature[i]);
							file.read(reinterpret_cast<char*> (m_instancesT[i].data()), static_cast<size_t>(m_sparseinstancesperfeature[i]) * sizeof(node));
						}
						else
						{
							auto type = m_features[i]->getFeature()->getType();
							file.read(reinterpret_cast<char*> (m_features[i]->getFeature()->getValues()), static_cast<size_t>(n) * sizeoftype[type]);
						}
					}
					if (Property::getProperty()->getDatasetType() == DataType::LibSVMCF || Property::getProperty()->getDatasetType() == DataType::LibSVMRG)
						offset += sizeof(node) * m_sparseinstancesperfeature[i];
					else
					{
						auto type = m_features[i]->getFeature()->getType();
						offset += sizeoftype[type];
					}
				}
			}
			else
			{
				file.seekg(offset, ios::beg);
				for (int i = 0; i < m_featuresize - 1; ++i)
				{
					if (Property::getProperty()->getDatasetType() == DataType::LibSVMCF || Property::getProperty()->getDatasetType() == DataType::LibSVMRG)
					{
						m_instancesT[i].resize(m_sparseinstancesperfeature[i]);
						file.read(reinterpret_cast<char*> (m_instancesT[i].data()), static_cast<size_t>(m_sparseinstancesperfeature[i]) * sizeof(node));
					}
					else {
						string name = m_features[i]->getName();
						const auto& feature = m_features[i]->getFeature();

						auto type = feature->getType();
						file.read(reinterpret_cast<char*> (feature->getValues()), static_cast<size_t>(n) * sizeoftype[type]);
					}
				}
			}
		}
		else
		{
			LOG(ERROR) << datafilepath << " open failed!";
			file.close();
			return false;
		}
		file.close();
		return true;
	}
	
	std::shared_timed_mutex* DataSet::getMutex() 
	{ 
		if (!m_rwmutex)
			m_rwmutex = new std::shared_timed_mutex; 
		return m_rwmutex; 
	}
	
	PFeatureInfo DataSet::getFeatureFromOld(const PFeatureInfo& featureinfo, bool needdata)
	{
		int featureindex = getFeatureIndex(featureinfo);
		return getFeature(featureindex, needdata);
	}
	
	void DataSet::UpdateNameFeature()
	{
		int i = 0;
		for (const auto& fea : m_features)
		{
			fea->setIndex(i++);
			//m_namefeatureindex[fea->getName()] = i++;
			if (fea->getType() == OutType::Discrete)
				m_discretefeaturesize++;
		}
	}

	void DataSet::getInstancesTfromThunder()
	{
		
		if (m_thunderdata == nullptr || m_thunderdata->n_features() == 0 || m_thunderdata->n_instances() == 0)
		{
			LOG(ERROR) << "DataSet getInstancesTfromThunder Error: m_thunderdata is nullptr or empty!";
			exit(1);
		}
		int m = 0, n = 0;
		
		node2d* data = const_cast<node2d*> (& m_thunderdata->instances());
		bool iszerobased = m_thunderdata->is_zero_based();
		m = data->size(), n = m_featuresize - 1;
		getT(data, &m_instancesT, n, m, iszerobased);
		if(m_thunderdata){delete m_thunderdata; m_thunderdata = nullptr;}
		
		LOG(INFO) << "DataSet getInstancesTfromThunder: m_instancesT size = " << m_instancesT.size();
	}

	PFeatureInfo DataSet::getFeatureFromInstanceT(const PFeatureInfo& featureinfo)
	{
		int featureindex = getFeatureIndex(featureinfo);
		return getFeatureFromInstanceT(featureindex);
	}

	PFeatureInfo DataSet::getFeatureFromInstanceT(int featureindex)
	{
		if (m_instancesT.size() == 0)
		{
			LOG(ERROR) << "DataSet getFeatureFromInstanceT Error: m_instancesT is empty!";
			exit(1);
		}
		auto &featurenode = m_instancesT[featureindex];
		auto &featureinfo = m_features[featureindex];
		Feature* feature = nullptr;
		switch (featureinfo->getType())
		{
			case OutType::Numeric:
				feature = new NumericFeature(m_instancesoffeature);
				break;
			default:
				feature = new DiscreteFeature(m_instancesoffeature);
		}
		for (int i = 0; i < featurenode.size(); ++i)
		{
			auto index = featurenode[i].index;
			auto value = featurenode[i].value;
			if (featureinfo->getType() == OutType::Numeric)
			{
				MyDataType value2 = static_cast<MyDataType>(value);
				feature->setValue(index, &value2);
			}
			else
			{
				int value2 = static_cast<int>(value);
				feature->setValue(index, &value2);
			}
		}
		featureinfo->setFeature(feature);
		return featureinfo;
	}

	void DataSet::setFeatureInstanceT(const PFeatureInfo& featureinfo)
	{
		if (m_instancesT.size() > 0 || m_thunderdata)
		{
			int featureindex = getFeatureIndex(featureinfo);
			setFeatureInstanceT(featureindex);
		}
	}

	void DataSet::setFeatureInstanceT(int featureindex)
	{
		if (m_instancesT.size() == 0)
		{
			LOG(ERROR) << "DataSet setFeatureInstanceT Error: m_instancesT is empty!";
			exit(1);
		}
		auto &featurenode = m_instancesT[featureindex];
		auto &featureinfo = m_features[featureindex];
		auto feature = featureinfo->getFeature();
		if (feature == nullptr)
		{
			LOG(ERROR) << "DataSet setFeatureInstanceT Error: feature is nullptr!";
			exit(1);
		}
		featurenode.clear();
		for (int i = 0; i < m_instancesoffeature; ++i)
		{
			if (feature->getType() == FeatureType::Numeric)
			{
				MyDataType value = *reinterpret_cast<MyDataType*>(feature->getValue(i));
				if (value != 0)
					featurenode.push_back(node(i, static_cast<float_type>(value)));
			}
			else if (feature->getType() == FeatureType::Discrete)
			{
				int value = *reinterpret_cast<int*>(feature->getValue(i));
				if (value != 0)
					featurenode.push_back(node(i, static_cast<float_type>(value)));
			}
		}
		m_sparseinstancesperfeature[featureindex] = static_cast<int>(featurenode.size());
	}

	void DataSet::updateDataSet()
	{
		if (m_selectedfeatures.size() > 0)
		{
			std::vector<PFeatureInfo> features;
			std::vector<bool> featureisused(m_featuresize-1, 0);
			for (auto& featureindex : m_selectedfeatures)
				featureisused[featureindex] = true;
			
			std::vector<std::shared_timed_mutex*> featurelock;
			std::vector<int> featureused;
			for (int i = 0;i < m_featuresize-1;++ i)
			{
				if (featureisused[i])
				{
					features.push_back(m_features[i]);
					featurelock.push_back(m_featurelock[i]);
					featureused.push_back(m_featureused[i]);
				}
				else
				{
					m_features[i]->clear();
					delete m_features[i];
					m_features[i] = nullptr;
					if(m_featurelock[i])
					{
						delete m_featurelock[i];
						m_featurelock[i] = nullptr;
					}
				}
			}
			if (m_thunderdata || m_instancesT.size())
			{
				// delete the elements in m_instancesT when featureisused is false
				// 使用 std::stable_partition 保持顺序不变
				auto it_instancesT = std::stable_partition(m_instancesT.begin(), m_instancesT.end(), [index = 0, &featureisused](const std::vector<node>&) mutable {
					return featureisused[index++];
					});

				auto it_sparseinstancesperfeature = std::stable_partition(m_sparseinstancesperfeature.begin(), m_sparseinstancesperfeature.end(), [index = 0, &featureisused](int) mutable {
					return featureisused[index++];
					});

				// 移除未使用的元素
				m_instancesT.erase(it_instancesT, m_instancesT.end());
				m_sparseinstancesperfeature.erase(it_sparseinstancesperfeature, m_sparseinstancesperfeature.end());

			}
			m_featuresize = features.size() + 1;
			m_features = features;
			m_featurelock = featurelock;
			m_featureused = featureused;
			m_discretefeaturesize = 0;
			
			for (auto& feature : m_features)
			{
				if (feature->getType() == OutType::Discrete)
					m_discretefeaturesize++;
			}

		}

	}

	void DataSet::LoadThunderData(string datapath, int classes, string dataname)
	{
		if (m_thunderdata)
		{
			delete m_thunderdata;
			m_thunderdata = nullptr;
		}
		m_name = dataname;
		m_thunderdata = new ThunderSVM::DataSet;
		m_thunderdata->load_from_file(datapath);
		m_thunderdata->group_classes(classes > 1);
		LOG(INFO) << "DataSet LoadThunderData: " << datapath << " load successful!";
		int numfeatures = m_thunderdata->n_features();
		if (Property::getProperty()->getFeatureNum() == 0)
			Property::getProperty()->setFeatureNum(numfeatures);
		else if (Property::getProperty()->getFeatureNum() != numfeatures)
		{
			LOG(WARNING) << "DataSet LoadThunderData: featurenum = " << numfeatures << " but given " << Property::getProperty()->getFeatureNum();
			numfeatures = Property::getProperty()->getFeatureNum();
		}
		for (int i = 0; i < numfeatures; ++i)
		{
			string name = std::to_string(i);
			//m_namefeatureindex[name] = i;
			PFeatureInfo featureinfo = new FeatureInfo(nullptr, std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Numeric, name);
			m_features.push_back(featureinfo);
		}
		m_featuresize = numfeatures + 1;
		m_instancesoffeature = m_thunderdata->n_instances();
		m_numofdatasetinstances.push_back(m_instancesoffeature);
		int trueclasses = m_thunderdata->n_classes();
		if (trueclasses != classes)
		{
			if(trueclasses > 1)
			{
				LOG(WARNING) << "DataSet LoadThunderData: classes = " << trueclasses << " but given " << classes;
				classes = trueclasses;
				Property::getProperty()->setTargetClasses(classes);
			}
		}
		Feature* targetfeature = nullptr;
		OutType outtype = OutType::Discrete;
		switch (Property::getProperty()->getDatasetType())
		{
			case DataType::LibSVMCF:
				targetfeature = new DiscreteFeature(m_instancesoffeature, classes);

				break;
			case DataType::LibSVMRG:
				targetfeature = new NumericFeature(m_instancesoffeature);
				outtype = OutType::Numeric;
				break;
		}
		std::unordered_map<int, int> keyindex;
		int classesindex = 1;
		for (int i = 0; i < m_instancesoffeature; ++i)
		{
			int value = static_cast<int>(m_thunderdata->y()[i]);
			if (keyindex[value] == 0)
				keyindex[value] = classesindex++;
		}
		for (int i = 0; i < m_instancesoffeature; ++i)
		{
			if (outtype == OutType::Discrete)
			{
				int value = keyindex[static_cast<int>(m_thunderdata->y()[i])] - 1;
				targetfeature->setValue(i, &value);
			}
			else if (outtype == OutType::Numeric)
			{
				MyDataType value = static_cast<MyDataType>(m_thunderdata->y()[i]);
				targetfeature->setValue(i, &value);
			}
		}
		m_targetfeature = new FeatureInfo(targetfeature, std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), outtype, "target", false, classes);
		Init();
		m_datasetindex[1] = m_instancesoffeature;
		if (Property::getProperty()->getDistributedNodes() > 1)
		{
#ifdef USE_MPICH
			int suminstances = 0;
			int process_id = 0, num_process = 0;
			MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
			MPI_Comm_size(MPI_COMM_WORLD, &num_process);
			int* counts = new int[num_process], * sides = new int[num_process];
			memset(counts, 0, num_process * sizeof(int));
			memset(sides, 0, num_process * sizeof(int));
			int instanceoffeature = m_instancesoffeature;
			int onevalsizes = sizeof(int);
			int senddata = instanceoffeature;


			MPI_Gather(&senddata, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if (process_id == 0)
			{
				for (int i = 0; i < num_process; ++i)
					suminstances += counts[i];
			}
			MPI_Bcast(&suminstances, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			m_distributedinstancesoffeature = suminstances;
#else
			LOG(ERROR) << "DataSet LoadThunderData Error: USE_MPICH not defined!";
			exit(1);
#endif
		}
		getInstancesTfromThunder();
	}

	void DataSet::LoadThunderDatas(string datapath, std::vector<string> datanames, int classes, string datasetname)
	{
		if (m_thunderdata)
		{
			delete m_thunderdata;
			m_thunderdata = nullptr;
		}
		m_name = datasetname;
		m_thunderdata = new ThunderSVM::DataSet;
		std::vector<float_type> targetfeature;
		std::vector<int> datasetindex{0};
		size_t maxdatasetclass = 0;
		m_numid = datanames.size();
		for (int i = 0; i < datanames.size(); ++i)
		{
			if(!m_thunderdata) m_thunderdata = new ThunderSVM::DataSet;
			m_thunderdata->load_from_file(datapath + datanames[i]);
			m_thunderdata->group_classes(classes > 0);
			LOG(INFO) << "DataSet LoadThunderDatas: " << datapath + datanames[i] << " load successful!";
			maxdatasetclass = std::max(maxdatasetclass, m_thunderdata->n_classes());
			if (m_features.size() == 0)
			{
				int numfeatures = m_thunderdata->n_features();
				if (Property::getProperty()->getFeatureNum() == 0)
					Property::getProperty()->setFeatureNum(numfeatures);
				else if (Property::getProperty()->getFeatureNum() != numfeatures)
				{
					LOG(WARNING) << "DataSet LoadThunderDatas: featurenum = " << numfeatures << " but given " << Property::getProperty()->getFeatureNum();
					numfeatures = Property::getProperty()->getFeatureNum();
				}
				for (int i = 0; i < numfeatures; ++i)
				{
					string name = std::to_string(i);
					//m_namefeatureindex[name] = i;
					PFeatureInfo featureinfo = new FeatureInfo(nullptr, std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Numeric, name);
					m_features.push_back(featureinfo);
				}
				m_featuresize = numfeatures + 1;
			}
			m_sparseinstancesperfeature.resize(m_featuresize - 1);
			m_instancesoffeature = m_thunderdata->n_instances();
			targetfeature.insert(targetfeature.end(), m_thunderdata->y().begin(), m_thunderdata->y().end());
			datasetindex.push_back(targetfeature.size());
			m_numofdatasetinstances.push_back(m_instancesoffeature);
			getInstancesTfromThunder();
			for (int i = 0; i < m_instancesT.size(); ++i)
				m_sparseinstancesperfeature[i] = m_instancesT[i].size();
			setID(i);
			serialize();
			if (i < datanames.size() - 1)
			{
				m_instancesT.clear();
				m_instancesT.shrink_to_fit();
			}
			
		}
		m_datasetindex = datasetindex;
		delete m_thunderdata;
		m_thunderdata = nullptr;
		if (static_cast<int>(maxdatasetclass) != classes)
		{
			if(maxdatasetclass > 1)
			{
				LOG(WARNING) << "DataSet LoadThunderDatas: classes = " << maxdatasetclass << " but given " << classes;
				classes = static_cast<int>(maxdatasetclass);
				Property::getProperty()->setTargetClasses(classes);
			}
		}
		Feature* ptargetfeature = nullptr;
		OutType outtype = OutType::Discrete;
		switch (Property::getProperty()->getDatasetType())
		{
		case DataType::LibSVMCF:
			ptargetfeature = new DiscreteFeature(getInstancesOfFeature(3), classes);

			break;
		case DataType::LibSVMRG:
			ptargetfeature = new NumericFeature(getInstancesOfFeature(3));
			outtype = OutType::Numeric;
			break;
		}
		std::unordered_map<int, int> keyindex;
		int classesindex = 1;
		for (int i = 0; i < getInstancesOfFeature(3); ++i)
		{
			int value = static_cast<int>(targetfeature[i]);
			if (keyindex[value] == 0)
				keyindex[value] = classesindex++;
		}
		for (int i = 0; i < getInstancesOfFeature(3); ++i)
		{
			if (outtype == OutType::Discrete)
			{
				int value = keyindex[static_cast<int>(targetfeature[i])] - 1;
				ptargetfeature->setValue(i, &value);
			}
			else if (outtype == OutType::Numeric)
			{
				MyDataType value = static_cast<MyDataType>(targetfeature[i]);
				ptargetfeature->setValue(i, &value);
			}
		}
		m_targetfeature = new FeatureInfo(ptargetfeature, std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), outtype, "target", false, classes);
		Init();
	}

	void DataSet::clearFeatureData(int featureindex)
	{
		
		m_featurelock[featureindex]->lock();
		m_featureused[featureindex]--;
		if (m_featureused[featureindex] <= 0)
		{
			auto &featureinfo = m_features[featureindex];
			auto feature = featureinfo->getFeature();
			if (feature) delete feature;
			featureinfo->setFeature(nullptr);
			m_featureusednum--;
			if (m_featureusednum < 0) m_featureusednum = 0;
		}
		
		m_featurelock[featureindex]->unlock();
	}

	void DataSet::clearFeatureData(const PFeatureInfo& featureinfo)
	{
		if (m_thunderdata ||  m_instancesT.size() > 0)
		{
			auto featureindex = getFeatureIndex(featureinfo);
			clearFeatureData(featureindex);
		}
	}

	int DataSet::getFeatureIndex(const PFeatureInfo & featureinfo)
	{
		/*if (m_namefeatureindex.find(name) == m_namefeatureindex.end())
		{
			LOG(ERROR) << "DataSet getFeatureIndex Error: " << name  << " not found!";
			exit(-1);
		}*/
		//return m_namefeatureindex[name];
		int index = featureinfo->getIndex();
		if (index < 0 || index >= m_featuresize)
		{
			LOG(ERROR) << "DataSet getFeatureIndex Error: index = " << index << " out of range!";
			exit(-1);
		}
		return index;
	}

	void DataSet::setFeatureFromOld(const PFeatureInfo& oldfeatureinfo, PFeatureInfo& featureinfo)
	{
		int featureindex = getFeatureIndex(oldfeatureinfo);
		setFeature(featureindex, featureinfo);
	}

	int DataSet::megerDataSet(DataSet* valdataset, int datatype)
	{
		if (valdataset == nullptr)
		{
			if (datatype == 0)
			{
				LOG(INFO) << "DataSet megerDataSet : valdataset is nullptr, split traindataset!";
				std::mt19937 rng(42);
				vector<int> index(m_instancesoffeature);
				for (int i = 0; i < m_instancesoffeature; ++i) index[i] = i;
				std::shuffle(index.begin(), index.end(), rng);
				int trainnum = static_cast<int>(m_instancesoffeature * 0.8);
				m_trainindex.resize(trainnum);
				for (int i = 0; i < trainnum; ++i) m_trainindex[i] = index[i];
				m_valindex.resize(m_instancesoffeature - trainnum);
				for (int i = trainnum; i < m_instancesoffeature; ++i) m_valindex[i - trainnum] = index[i];
				std::sort(m_trainindex.begin(), m_trainindex.end());
				std::sort(m_valindex.begin(), m_valindex.end());
				m_hasval = false;
				return 1;
			}
			else
			{
				LOG(WARNING) << "DataSet megerDataSet WARNING: testdataset is nullptr!";
				return 1;
			}
		}
		if (m_featuresize != valdataset->getFeatureSize(true))
		{
			LOG(ERROR) << "DataSet megerDataSet Error: m_featuresize != valdataset->getFeatureSize()!";
			return -1;
		}
		vector<int> *pindex = nullptr;
		if (datatype == 0) pindex = &m_valindex, m_hasval = true;
		else pindex = &m_testindex, m_hastest = true;
		if (m_trainindex.size() == 0)
		{
			m_trainindex.resize(m_instancesoffeature);
			for (int i = 0; i < m_instancesoffeature; ++i) m_trainindex[i] = i;
		}
		pindex->resize(valdataset->getInstancesOfFeature());
		for (int i = 0; i < valdataset->getInstancesOfFeature(); ++i) pindex->at(i) = m_instancesoffeature + i;
		int totalinstances = m_instancesoffeature + valdataset->getInstancesOfFeature();
		m_instancesoffeature = totalinstances;
		m_datasetindex[m_numid] = totalinstances;

		for (int i = 0; i < m_featuresize - 1; ++i)
		{
			if (m_features[i]->getName() != valdataset->getFeature(i)->getName())
			{
				LOG(ERROR) << "DataSet megerDataSet Error, Feature name: " << m_features[i]->getName() << " != " << valdataset->getFeature(i)->getName();
				return -1;
			}
			if (!m_features[i]->getFeature())m_features[i] = getFeatureFromOld(m_features[i], true);
			auto feature = m_features[i]->getFeature();
			auto valfeatureinfos = valdataset->getFeatures();
			if (valfeatureinfos[i]->getFeature() == nullptr)valfeatureinfos[i] = getFeatureFromOld(valfeatureinfos[i], true);
			auto valfeature = valfeatureinfos[i]->getFeature();
			if (feature->getType() != valfeature->getType())
			{
				LOG(ERROR) << "DataSet megerDataSet Error, Feature type: " << m_features[i]->getName() << " != " << valfeatureinfos[i]->getName();
				return -1;
			}
			feature->merge(valfeature, valdataset->getInstancesOfFeature());
			m_features[i]->setFeature(feature);
			setFeatureInstanceT(m_features[i]);
			clearFeatureData(valfeatureinfos[i]);
		}
		if (m_targetfeature->getName() != valdataset->getTargetFeature()->getName())
		{
			LOG(ERROR) << "DataSet megerDataSet Error, TargetFeature name: " << m_targetfeature->getName() << " != " << valdataset->getTargetFeature()->getName();
			return -1;
		}
		auto targetfeature = m_targetfeature->getFeature();
		auto valtargetfeature = valdataset->getTargetFeature()->getFeature();
		if (targetfeature->getType() != valtargetfeature->getType())
		{
			LOG(ERROR) << "DataSet megerDataSet Error, TargetFeature type: " << m_targetfeature->getName() << " != " << valdataset->getTargetFeature()->getName();
			return -1;
		}
		targetfeature->merge(valtargetfeature, valdataset->getInstancesOfFeature());
		m_targetfeature->setFeature(targetfeature);
		return 1;
	}

	TempDataSet::TempDataSet() :m_index(0), m_rwmutex(nullptr) {}
	
	TempDataSet::TempDataSet(const std::vector<DataSet* >& tempdataset) :m_index(0), m_rwmutex(nullptr) { m_tempdataset = tempdataset; getMutex(); }
	
	TempDataSet::~TempDataSet() { Release(); }
	
	DataSet* TempDataSet::getNext() { m_index = ++m_index % m_tempdataset.size(); return getDataSet(); }
	
	int TempDataSet::getIndex() { return m_index; }
	
	std::vector<DataSet*> TempDataSet::getDataSets()
	{
		return m_tempdataset;
	}
	
	void TempDataSet::Init(const std::vector<DataSet* >& tempdataset)
	{
		m_tempdataset = tempdataset;
		m_index = 0;
		getMutex();
	}
	
	void TempDataSet::Release()
	{
		m_tempdataset.clear();
		delete m_rwmutex;
		m_rwmutex = nullptr;
	}
	
	DataSet* TempDataSet::getDataSet()
	{
		if (m_index >= m_tempdataset.size())
		{
			LOG(ERROR) << "TempDataSet getDataSet Error: out of vector!";
			return nullptr;
		}
		else return m_tempdataset[m_index];
	}
	
	std::shared_timed_mutex* TempDataSet::getMutex()
	{
		if (!m_rwmutex)
			m_rwmutex = new  std::shared_timed_mutex;
		return  m_rwmutex;
	}

}//RL4FS