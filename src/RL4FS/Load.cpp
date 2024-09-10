#include "RL4FS/Load.h"
namespace RL4FS
{
	bool Load::StdDataSetIndex(DataSet* dataset, PFeatureInfo& featureinfo, int index)
	{
		if (featureinfo->getType() != OutType::Discrete && featureinfo->getType() != OutType::String)
			return false;

		int discretevalue = 0;
		std::unordered_map<string, int> mapdiscrete;
		int maxinstanceoffeature = 0;
		string missval = Property::getProperty()->getMissVal();
		for (int dataid = 0; dataid < dataset->getNumID(); ++dataid)
		{
			dataset->setID(dataid);
			dataset->deserialize();
			maxinstanceoffeature = std::max(maxinstanceoffeature, dataset->getInstancesOfFeature());
			auto feature = featureinfo->getFeature();
			for (int i = 0; i < dataset->getInstancesOfFeature(); ++i)
			{
				string val = *reinterpret_cast<string*>(feature->getValue(i));
				if (val == missval) mapdiscrete[val] = 0;
				else {
					if (mapdiscrete.find(val) == mapdiscrete.end())
						mapdiscrete[val] = discretevalue++;
				}
			}
		}

		Feature* newfeature = new DiscreteFeature(maxinstanceoffeature, discretevalue);
		auto oldfeature = featureinfo->getFeature();
		for (int dataid = 0; dataid < dataset->getNumID(); ++dataid)
		{
			dataset->setID(dataid);
			dataset->deserialize();
			auto feature = featureinfo->getFeature();

			int* val = new int;
			for (int i = 0; i < dataset->getInstancesOfFeature(); ++i)
			{
				*val = mapdiscrete[*reinterpret_cast<string*>(feature->getValue(i))];
				newfeature->setValue(i, reinterpret_cast<void*>(val));
			}
			delete val;
			featureinfo->setFeature(newfeature);
			dataset->serialize();
			featureinfo->setFeature(feature);
		}
		delete oldfeature;
		oldfeature = nullptr;
		featureinfo->setFeature(newfeature);
		featureinfo->setNumsOfValues(discretevalue);
		newfeature = nullptr;
		return true;
	}
	
	bool Load::StdDataSet(DataSet* dataset, bool ischange, int index)
	{
		if (index >= 0)
		{
			if (index >= dataset->getFeatureSize(false))
			{
				LOG(ERROR) << "StdDataSet Index out of features's size!\n";
				return false;
			}
			auto featureinfo = dataset->getFeature(index);
			if (featureinfo->getType() != OutType::Discrete && featureinfo->getType() != OutType::String)
				return false;
			StdDataSetIndex(dataset, featureinfo, index);
		}
		else if (index == -1)
		{
			const int m = dataset->getFeatureSize();

			for (int post = 0; post < m; ++post)
			{
				StdDataSet(dataset, ischange, post);
			}

			auto featureinfo = dataset->getTargetFeature();
			if (featureinfo->getType() == OutType::Discrete || featureinfo->getType() == OutType::String)
			{
				auto feature = featureinfo->getFeature();
				auto instancesoffeature = dataset->getDatasetIndex()[dataset->getNumID()];
				std::set<string> valdiscrete;
				for (int i = 0; i < instancesoffeature; ++i)
					valdiscrete.insert(*reinterpret_cast<string*>(feature->getValue(i)));
				int discretevalue = 0;
				std::unordered_map<string, int> mapdiscrete;
				for (auto& i : valdiscrete)
					mapdiscrete[i] = discretevalue++;
				Feature* newfeature = new DiscreteFeature(instancesoffeature, discretevalue);
				for (int i = 0; i < instancesoffeature; ++i)
				{
					int* val = new int;
					*val = mapdiscrete[*reinterpret_cast<string*>(feature->getValue(i))];
					newfeature->setValue(i, reinterpret_cast<void*>(val));
					delete val;
				}
				delete feature;
				featureinfo->setFeature(newfeature);
				featureinfo->setNumsOfValues(discretevalue);
				if (ischange)
				{
					if (discretevalue != Property::getProperty()->getTargetClasses())
						LOG(WARNING) << "TargetClasses Num is " << discretevalue << " , but the default is " << Property::getProperty()->getTargetClasses() << "!";
					Property::getProperty()->setTargetClasses(discretevalue);
				}
				newfeature = nullptr;
			}
		}
		return true;
	}

	bool Load::DistributedStdDataSetIndex(DataSet* dataset, PFeatureInfo& featureinfo, bool ischange)
	{
#ifdef USE_MPICH
		if (featureinfo->getType() != OutType::Discrete && featureinfo->getType() != OutType::String)
			return 0;

		int discretevalue = 0;
		std::unordered_map<string, int> mapdiscrete;
		int process_id = 0, num_process = 0;
		MPI_Comm_size(MPI_COMM_WORLD, &num_process);
		MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

		int* counts = new int[num_process], * counts2 = new int[num_process];
		memset(counts, 0, num_process * sizeof(int));
		memset(counts2, 0, num_process * sizeof(int));
		int* sides = new int[num_process], * sides2 = new int[num_process];
		memset(sides2, 0, num_process * sizeof(int));
		memset(sides, 0, num_process * sizeof(int));
		int instanceoffeature = dataset->getInstancesOfFeature();
		int onevalsizes = sizeof(std::string);
		int senddata = instanceoffeature * onevalsizes;

		MPI_Gather(&senddata, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
		int post = 0;
		for (int i = 0; i < num_process; ++i)
			sides[i] = post, post += counts[i];
		if (process_id) post = 1;
		char* newfeaturestr = new char[post];
		char* startfeaturestr = reinterpret_cast<char*>(featureinfo->getFeature()->getValues());

		MPI_Gatherv(startfeaturestr, senddata, MPI_CHAR, newfeaturestr, counts, sides, MPI_CHAR, 0, MPI_COMM_WORLD);

		int sendrealdata = 0;
		for (int i = 0; i < instanceoffeature; ++i)
		{
			string* val = reinterpret_cast<string*>(featureinfo->getFeature()->getValue(i));
			int length = val->size();
			sendrealdata += length;
		}
		char* sendrealdatastr = new char[sendrealdata];
		memset(sendrealdatastr, 0, sendrealdata * sizeof(char));
		for (int i = 0, lengthindex = 0; i < instanceoffeature; ++i)
		{
			string* val = reinterpret_cast<string*>(featureinfo->getFeature()->getValue(i));
			int length = val->size();
			memcpy(sendrealdatastr + lengthindex, val->c_str(), length);
			lengthindex += length;
		}
		MPI_Gather(&sendrealdata, 1, MPI_INT, counts2, 1, MPI_INT, 0, MPI_COMM_WORLD);
		int post2 = 0;
		for (int i = 0; i < num_process; ++i)
			sides2[i] = post2, post2 += counts2[i];
		if (process_id) post2 = 1;
		char* reciverealdatastr = new char[post2];
		MPI_Gatherv(sendrealdatastr, sendrealdata, MPI_CHAR, reciverealdatastr, counts2, sides2, MPI_CHAR, 0, MPI_COMM_WORLD);



		char* newfeaturesendstr = nullptr;
		if (!process_id)
		{ 
			for (int i = 0, lengthindex = 0; i < post / onevalsizes; ++i)
			{
				string* valtemp = reinterpret_cast<string*>(newfeaturestr + i * onevalsizes);
				string val = string(reciverealdatastr + lengthindex, valtemp->size());
				if (val == Property::getProperty()->getMissVal()) mapdiscrete[val] = 0;
				else {
					if (mapdiscrete.find(val) == mapdiscrete.end())
						mapdiscrete[val] = discretevalue++;
				}
				lengthindex += valtemp->size();
			}
		}

		int suminstances = post / onevalsizes;
		MPI_Bcast(&discretevalue, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&suminstances, 1, MPI_INT, 0, MPI_COMM_WORLD);
		dataset->setDistributedInstancesofFeature(suminstances);
		Feature* allfeature = new DiscreteFeature(post / onevalsizes, discretevalue);
		Feature* newfeature = new DiscreteFeature(instanceoffeature, discretevalue);
		int* val = new int;
		if (!process_id)
		{
			for (int i = 0, lengthindex = 0; i < post / onevalsizes; ++i)
			{
				string* sval = reinterpret_cast<string*>(newfeaturestr + i * onevalsizes);
				*val = mapdiscrete[string(reciverealdatastr + lengthindex, sval->size())];
				lengthindex += sval->size();
				allfeature->setValue(i, reinterpret_cast<void*>(val));
			}

			newfeaturesendstr = reinterpret_cast<char*>(allfeature->getValues());
			for (int i = 0; i < num_process; ++i)
				counts[i] /= onevalsizes, sides[i] /= onevalsizes;
		}

		delete[] newfeaturestr;
		delete[] reciverealdatastr;
		delete[] sendrealdatastr;
		delete[] counts2;
		delete[] sides2;

		int* recivestr = new int[instanceoffeature];
		memset(recivestr, 0, instanceoffeature * sizeof(int));

		MPI_Scatterv(newfeaturesendstr, counts, sides, MPI_INT, recivestr, instanceoffeature, MPI_INT, 0, MPI_COMM_WORLD);
		for (int i = 0; i < instanceoffeature; ++i)
		{
			*val = *(recivestr + i);
			newfeature->setValue(i, reinterpret_cast<void*>(val));
		}

		delete[] counts;
		delete[] sides;
		delete allfeature;
		delete[] recivestr;
		delete val;
		auto feature = featureinfo->getFeature();
		delete feature;
		feature = nullptr;
		featureinfo->setFeature(newfeature);
		featureinfo->setNumsOfValues(discretevalue);
		newfeature = nullptr;

		if (ischange)
		{
			if (discretevalue != Property::getProperty()->getTargetClasses())
				LOG(WARNING) << "TargetClasses Num is " << discretevalue << " , but the default is " << Property::getProperty()->getTargetClasses() << "!";
			Property::getProperty()->setTargetClasses(discretevalue);
		}
		return true;
#else
		LOG(ERROR) << "loadData Please USE_MPICH = ON when CMake this project, or you cannot set Property::getProperty()->setDistributedNodes() > 1";
		exit(-1);
#endif
	}

	bool Load::DistributedStdDataSet(DataSet* dataset, bool ischange, int index)
	{
		if (index >= 0)
		{
			if (index >= dataset->getFeatureSize(false))
			{
				LOG(ERROR) << "StdDataSet Index out of features's size!\n";
				return false;
			}
			auto featureinfo = dataset->getFeature(index);
			if (featureinfo->getType() != OutType::Discrete && featureinfo->getType() != OutType::String)
				return false;
			DistributedStdDataSetIndex(dataset, featureinfo);
		}
		else
		{
			auto featuresize = dataset->getFeatureSize();
			for (int i = 0; i < featuresize; ++i)
			{
				auto featureinfos = dataset->getFeatures();
				auto featype = featureinfos[i]->getType();
				if (featype != OutType::Discrete && featype != OutType::String)continue;
				DistributedStdDataSet(dataset, ischange, i);
			}
			auto featureinfo = dataset->getTargetFeature();
			if (featureinfo->getType() == OutType::Discrete || featureinfo->getType() == OutType::String)
			{
				DistributedStdDataSetIndex(dataset, featureinfo, true);
			}
		}
		return true;
	}

	inline void printPoints(DataSet*& dataset)
	{
		dataset->updateNumPoints();
		auto numpoints = dataset->getNumPoints();
		string strnumpoints;
		for (const auto& i : numpoints)
			strnumpoints += std::to_string(i) + " ";
		int type = 1;
		if(dataset->getNumID() > 1)
			type = 3;
		LOG(INFO) << "Load dataset from " << dataset->getName() << " success, and the nums of point is: " << strnumpoints << " features: " << dataset->getFeatureSize(false) << " instances: " << dataset->getInstancesOfFeature(type);
	}

	DataSet* Load::loadData(string datapath, string classname, std::vector<string> discretefeaturename, std::vector<string> datefeaturename, int classes, string datasetname, bool hastitlee, bool ischange, bool loadtraindata, bool testdata)
	{
		if (ischange)
			Property::getProperty()->setDatasetName(datasetname)->setClassName(classname)->setTargetClasses(classes);
		
		std::vector<string> datapathfiles;

		GetFileNames(datapath, datapathfiles);

		int numdatasets = datapathfiles.size();
		if (numdatasets > 1 && Property::getProperty()->getDistributedNodes() > 1)
		{
			LOG(ERROR) << "Distribute can not exist with Chunk data set";
			return nullptr;
		}
		auto datapathtemp = datapath;
		if (numdatasets)
			datapath = datapathtemp + datapathfiles[0];
		int process_id = 0, num_process = 0;
		if (Property::getProperty()->getDistributedNodes() > 1 && loadtraindata)
		{
#ifdef USE_MPICH
			int provided;
			if (!testdata)
				MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

			MPI_Comm_size(MPI_COMM_WORLD, &num_process);
			MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
			auto testdatanums = Property::getProperty()->getNumTempDatasets();
			if(testdata && process_id >= testdatanums) return nullptr;
			datapath = datapath + std::to_string(process_id);
			Property::getProperty()->setDatasetName(datasetname + std::to_string(process_id));
#else
			LOG(ERROR) << "loadData Please USE_MPICH = ON when CMake this project, or you cannot set Property::getProperty()->setDistributedNodes() > 1";
			exit(-1);
#endif // USE_MPICH
		}
		if (Property::getProperty()->getDatasetType() == DataType::LibSVMCF || Property::getProperty()->getDatasetType() == DataType::LibSVMRG)
		{
			auto dataset = new DataSet();
			if (numdatasets)
				dataset->LoadThunderDatas(datapathtemp, datapathfiles, classes, datasetname);
			else
				dataset->LoadThunderData(datapath, classes, datasetname);
			printPoints(dataset);
			return dataset;
		}


		std::unordered_map<string, int> mapdiscretefeaturename;
		for (auto& name : discretefeaturename)
			mapdiscretefeaturename[name] = 1;
		for (auto& name : datefeaturename)
			mapdiscretefeaturename[name] = 2;

		using namespace csv2;
		csv2::Reader<delimiter<','>,
			quote_character<'"'>,
			first_row_is_header<true>,
			trim_policy::trim_whitespace> csv;
		int n = 0;
		int m = 0;
		int targetn = 0;
		int targetcolindex = 0;
		std::vector<PFeatureInfo> featureinfos;
		std::vector<Feature*> features;
		std::vector<string> colnames;
		Feature* targetfeature = nullptr;
		
		if (csv.mmap(datapath))
		{
			const auto header = csv.header();
			int i = 0;
			n = csv.rows() - 1;
			m = csv.cols();
			if (numdatasets) targetn = n * numdatasets;
			else targetn = n;
			for (const auto& cell : header)
			{
				string name;
				cell.read_value(name);
				auto indexr = name.find('\r');
				if (indexr != -1)name = name.substr(0, indexr);
				if (name == classname || name == classname + "\r")
				{
					targetcolindex = i;
					if (classes > 0)
						targetfeature = new StringFeature(targetn);
					else targetfeature = new NumericFeature(targetn);
					continue;
				}
				if (find(discretefeaturename.begin(), discretefeaturename.end(), name) != discretefeaturename.end())
					features.push_back(new StringFeature(n));
				else if (find(datefeaturename.begin(), datefeaturename.end(), name) != datefeaturename.end())
					features.push_back(new DateFeature(n));
				else features.push_back(new NumericFeature(n));
				std::replace(name.begin(), name.end(), ' ', '_');
				colnames.push_back(std::move(name));

				i++;
			}

			i = 0;
			for (const auto& row : csv) {
				int j = 0, featureindex = 0;
				for (const auto& cell : row) {
					// Do something with cell value
					std::string value;
					cell.read_value(value);
					if (j == targetcolindex)
					{
						if (classes)
						{
							string* val = new string;
							*val = std::move(value);
							std::replace((*val).begin(), (*val).end(), ' ', '_');
							targetfeature->setValue(i, reinterpret_cast<void*>(val));
							delete val;
						}
						else
						{
							MyDataType* val2 = new MyDataType;
							try {
								if (sizeof(MyDataType) == sizeof(float))
									*val2 = stof(value);
								else if (sizeof(MyDataType) == sizeof(double))
									*val2 = stod(value);
							}
							catch (...) {
								*val2 = 0.0;
							}
							targetfeature->setValue(i, reinterpret_cast<void*>(val2));
							delete val2;
						}

						featureindex--;
					}
					else
					{
						if (features[featureindex]->getType() == FeatureType::Discrete || features[featureindex]->getType() == FeatureType::String)
						{
							string* val = new string;
							*val = std::move(value);
							std::replace((*val).begin(), (*val).end(), ' ', '_');
							features[featureindex]->setValue(i, reinterpret_cast<void*>(val));
							delete val;
						}
						else if (features[featureindex]->getType() == FeatureType::Date)
						{
							Date* val = new Date;
							*val = Date(value);
							features[featureindex]->setValue(i, reinterpret_cast<void*>(val));
							delete val;
						}
						else
						{
							MyDataType* val = new MyDataType;
							try {
								if (sizeof (MyDataType) == sizeof (float))
									*val = stof(value);
								else if (sizeof (MyDataType) == sizeof (double))
									*val = stod(value);
							}
							catch (...) {
								if (sizeof (MyDataType) == sizeof (float))
									*val = 0.0F;
								else if (sizeof (MyDataType) == sizeof (double))
									*val = static_cast<double>(0.0);
							}
							features[featureindex]->setValue(i, reinterpret_cast<void*>(val));
							delete val;
						}
					}
					j++;
					featureindex++;
				}
				i++;
			}
		}
		else
		{
			LOG(ERROR) << "Load read csv error: " << datapath;
			return nullptr;
		}
		for (int i = 0; i < features.size(); ++i)
		{
			auto& name = colnames[i];
			if (features[i]->getType() == FeatureType::Discrete)
				featureinfos.emplace_back(new FeatureInfo(features[i], std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Discrete, name, false));
			else if (features[i]->getType() == FeatureType::Date)
				featureinfos.emplace_back(new FeatureInfo(features[i], std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Date, name, false));
			else if (features[i]->getType() == FeatureType::String)
				featureinfos.emplace_back(new FeatureInfo(features[i], std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Discrete, name, false));
			else
				featureinfos.emplace_back(new FeatureInfo(features[i], std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Numeric, name, false));
		}
		PFeatureInfo targetfeatureinfo = nullptr;
		if (classes)
			targetfeatureinfo = new FeatureInfo(targetfeature, std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Discrete, classname, false, classes);
		else
			targetfeatureinfo = new FeatureInfo(targetfeature, std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Numeric, classname, false);

		if (numdatasets == 0)
		{
			DataSet* dataset = new DataSet(featureinfos, targetfeatureinfo, datasetname, n);
			std::vector<int> datasetindex(2, 0);
			datasetindex[1] = dataset->getInstancesOfFeature();
			dataset->setDatasetIndex(datasetindex);
			dataset->relloc();
			if (Property::getProperty()->getDistributedNodes() > 1 && loadtraindata)
			{
				DistributedStdDataSet(dataset, ischange);
			}
			else
			{
				StdDataSet(dataset, ischange);
			}
			printPoints(dataset);
			return dataset;
		}
		else
		{
			std::vector<int> datasetindex(numdatasets + 1, 0);
			int targetindex = 0;
			DataSet* dataset = new DataSet(featureinfos, targetfeatureinfo, datasetname, n, 0, numdatasets);
			dataset->relloc();
			auto maxinstanceoffeature = dataset->getInstancesOfFeature();
			if (numdatasets > 1)dataset->serialize();
			csv2::Reader<delimiter<','>,
				quote_character<'"'>,
				first_row_is_header<false>,
				trim_policy::trim_whitespace> csvnohead;
			for (int datai = 1; datai < numdatasets; ++datai)
			{
				targetindex += dataset->getInstancesOfFeature();
				datasetindex[datai] = targetindex;
				datapath = datapathtemp + datapathfiles[datai];
				featureinfos = dataset->getFeatures();
				targetfeatureinfo = dataset->getTargetFeature();
				targetfeature = targetfeatureinfo->getFeature();
				if (csvnohead.mmap(datapath)) {
					//const auto header = csvnohead.header();
					int i = 0;
					n = csvnohead.rows();
					m = csvnohead.cols();

					i = 0;
					bool headflag = false;
					if (Property::getProperty()->getOtherDatasetHasHead())
						headflag = true;
					for (const auto& row : csvnohead) {
						int j = 0, featureindex = 0;
						if (headflag) { headflag = false; n--; continue; }
						if (n > maxinstanceoffeature)
						{
							LOG(ERROR) << "Load Dataset Error: Please use the most rows of dataset as the first dataset!";
							exit(-1);
						}
						for (const auto& cell : row) {
							// Do something with cell value
							std::string value;
							cell.read_value(value);
							if (j == targetcolindex)
							{

								if (classes)
								{
									string* val = new string;
									*val = std::move(value);
									std::replace((*val).begin(), (*val).end(), ' ', '_');
									targetfeature->setValue(i + targetindex, reinterpret_cast<void*>(val));
									delete val;
								}
								else
								{
									MyDataType* val2 = new MyDataType;
									try {
										if (sizeof(MyDataType) == sizeof(float))
											*val2 = stof(value);
										else if (sizeof(MyDataType) == sizeof(double))
											*val2 = stod(value);
									}
									catch (...) {
										*val2 = 0.0;
									}
									targetfeature->setValue(i + targetindex, reinterpret_cast<void*>(val2));
									delete val2;
								}

								featureindex--;
							}
							else
							{
								auto feature = featureinfos[featureindex]->getFeature();
								if (feature->getType() == FeatureType::Discrete || feature->getType() == FeatureType::String)
								{
									string* val = new string;
									*val = std::move(value);
									std::replace((*val).begin(), (*val).end(), ' ', '_');
									feature->setValue(i, reinterpret_cast<void*>(val));
									delete val;
								}
								else if (feature->getType() == FeatureType::Date)
								{
									Date* val = new Date;
									*val = Date(value);
									feature->setValue(i, reinterpret_cast<void*>(val));
									delete val;
								}
								else
								{
									MyDataType* val = new MyDataType;
									try {
										if (sizeof(MyDataType) == sizeof(float))
											*val = stof(value);
										else if (sizeof(MyDataType) == sizeof(double))
											*val = stod(value);
									}
									catch (...) {
										if (sizeof(MyDataType) == sizeof(float))
											*val = NAN;
										else if (sizeof(MyDataType) == sizeof(double))
											*val = static_cast<double>(NAN);
									}
									feature->setValue(i, reinterpret_cast<void*>(val));
									delete val;
								}
							}
							j++;
							featureindex++;
						}
						i++;
					}
				}
				dataset->setID(datai);
				dataset->setInstancesOfFeature(n);
				dataset->serialize();
			}
			datasetindex[numdatasets] = targetindex + dataset->getInstancesOfFeature();
			dataset->setDatasetIndex(datasetindex);
			StdDataSet(dataset, ischange);
			printPoints(dataset);
			return dataset;
		}
	}
	
	DataSet* Load::loadDataAllNumeber(string datapath, string classname, std::vector<string> discretefeaturename, std::vector<string> datefeaturename, int classes, string datasetname, bool hastitlee, bool ischange, bool loadtraindata, bool testdata)
	{
		if (ischange)
			Property::getProperty()->setDatasetName(datasetname)->setClassName(classname)->setTargetClasses(classes);
		
		std::vector<string> datapathfiles;

		GetFileNames(datapath, datapathfiles);

		int numdatasets = datapathfiles.size();
		if (numdatasets > 1 && Property::getProperty()->getDistributedNodes() > 1)
		{
			LOG(ERROR) << "Distribute can not exist with Chunk data set";
			return nullptr;
		}
		auto datapathtemp = datapath;
		if (numdatasets)
			datapath = datapathtemp + datapathfiles[0];
		int process_id = 0, num_process = 0;
		if (Property::getProperty()->getDistributedNodes() > 1 && loadtraindata)
		{
#ifdef USE_MPICH
			int provided;
			if (!testdata)
				MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

			MPI_Comm_size(MPI_COMM_WORLD, &num_process);
			MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
			auto testdatanums = Property::getProperty()->getNumTempDatasets();
			if(testdata && process_id >= testdatanums) return nullptr;
			datapath = datapath + std::to_string(process_id);
			Property::getProperty()->setDatasetName(datasetname + std::to_string(process_id));
#else
			LOG(ERROR) << "loadData Please USE_MPICH = ON when CMake this project, or you cannot set Property::getProperty()->setDistributedNodes() > 1";
			exit(-1);
#endif // USE_MPICH
		}
		if (Property::getProperty()->getDatasetType() == DataType::LibSVMCF || Property::getProperty()->getDatasetType() == DataType::LibSVMRG)
		{
			auto dataset = new DataSet();
			if (numdatasets)
				dataset->LoadThunderDatas(datapathtemp, datapathfiles, classes, datasetname);
			else
				dataset->LoadThunderData(datapath, classes, datasetname);
			printPoints(dataset);
			return dataset;
		}


		std::unordered_map<string, int> mapdiscretefeaturename;
		for (auto& name : discretefeaturename)
			mapdiscretefeaturename[name] = 1;
		for (auto& name : datefeaturename)
			mapdiscretefeaturename[name] = 2;

		using namespace csv2;
		csv2::Reader<delimiter<','>,
			quote_character<'"'>,
			first_row_is_header<true>,
			trim_policy::trim_whitespace> csv;
		int n = 0;
		int m = 0;
		int targetn = 0;
		int targetcolindex = 0;
		std::vector<PFeatureInfo> featureinfos;
		std::vector<Feature*> features;
		std::vector<string> colnames;
		Feature* targetfeature = nullptr;
		vector<int> featuresbins;
		std::set<int> setdiscrete;
		if (csv.mmap(datapath))
		{
			const auto header = csv.header();
			int i = 0;
			n = csv.rows() - 1;
			m = csv.cols();
			if (numdatasets) targetn = n * numdatasets;
			else targetn = n;
			for (const auto& cell : header)
			{
				string name;
				cell.read_value(name);
				auto indexr = name.find('\r');
				if (indexr != -1)name = name.substr(0, indexr);
				if (name == classname || name == classname + "\r")
				{
					targetcolindex = i;
					if (classes >= 2)
						targetfeature = new DiscreteFeature(targetn, classes);
					else targetfeature = new NumericFeature(targetn);
					continue;
				}
				if (find(discretefeaturename.begin(), discretefeaturename.end(), name) != discretefeaturename.end())
					features.push_back(new DiscreteFeature(n));
				else if (find(datefeaturename.begin(), datefeaturename.end(), name) != datefeaturename.end())
					features.push_back(new DateFeature(n));
				else features.push_back(new NumericFeature(n));
				std::replace(name.begin(), name.end(), ' ', '_');
				colnames.push_back(std::move(name));
				i++;
			}
			featuresbins.resize(features.size(), -1);
			i = 0;
			for (const auto& row : csv) {
				int j = 0, featureindex = 0;
				for (const auto& cell : row) {
					// Do something with cell value
					std::string value;
					cell.read_value(value);
					if (j == targetcolindex)
					{
						if (classes >= 2)
						{
							int* val = new int;
							*val = std::stoi(value);
							setdiscrete.insert(*val);
							targetfeature->setValue(i, reinterpret_cast<void*>(val));
							delete val;
						}
						else
						{
							MyDataType* val2 = new MyDataType;
							try {
								if (sizeof(MyDataType) == sizeof(float))
									*val2 = stof(value);
								else if (sizeof(MyDataType) == sizeof(double))
									*val2 = stod(value);
							}
							catch (...) {
								*val2 = 0.0;
							}
							targetfeature->setValue(i, reinterpret_cast<void*>(val2));
							delete val2;
						}

						featureindex--;
					}
					else
					{
						if (features[featureindex]->getType() == FeatureType::Discrete)
						{
							int* val = new int;
							*val = std::stoi(value);
							featuresbins[featureindex] = std::max(featuresbins[featureindex], *val);
							features[featureindex]->setValue(i, reinterpret_cast<void*>(val));
							delete val;
						}
						else if (features[featureindex]->getType() == FeatureType::Date)
						{
							Date* val = new Date;
							*val = Date(value);
							features[featureindex]->setValue(i, reinterpret_cast<void*>(val));
							delete val;
						}
						else
						{
							MyDataType* val = new MyDataType;
							try {
								if (sizeof (MyDataType) == sizeof (float))
									*val = stof(value);
								else if (sizeof (MyDataType) == sizeof (double))
									*val = stod(value);
							}
							catch (...) {
								if (sizeof (MyDataType) == sizeof (float))
									*val = 0.0F;
								else if (sizeof (MyDataType) == sizeof (double))
									*val = static_cast<double>(0.0);
							}
							features[featureindex]->setValue(i, reinterpret_cast<void*>(val));
							delete val;
						}
					}
					j++;
					featureindex++;
				}
				i++;
			}
		}
		else
		{
			LOG(ERROR) << "Load read csv error: " << datapath;
			return nullptr;
		}
		for (int i = 0; i < features.size(); ++i)
		{
			auto& name = colnames[i];
			if (features[i]->getType() == FeatureType::Discrete)
				featureinfos.emplace_back(new FeatureInfo(features[i], std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Discrete, name, false, featuresbins[i] + 1));
			else if (features[i]->getType() == FeatureType::Date)
				featureinfos.emplace_back(new FeatureInfo(features[i], std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Date, name, false));
			else if (features[i]->getType() == FeatureType::String)
				featureinfos.emplace_back(new FeatureInfo(features[i], std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Discrete, name, false));
			else
				featureinfos.emplace_back(new FeatureInfo(features[i], std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Numeric, name, false));
		}
		PFeatureInfo targetfeatureinfo = nullptr;
		
		if (classes >= 2)
			targetfeatureinfo = new FeatureInfo(targetfeature, std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Discrete, classname, false, classes);
		else
			targetfeatureinfo = new FeatureInfo(targetfeature, std::vector<PFeatureInfo>(), std::vector<PFeatureInfo>(), OutType::Numeric, classname, false);
		
		
		DataSet* dataset = nullptr;
		if (numdatasets == 0)
		{
			dataset = new DataSet(featureinfos, targetfeatureinfo, datasetname, n);
			std::vector<int> datasetindex(2, 0);
			datasetindex[1] = dataset->getInstancesOfFeature();
			dataset->setDatasetIndex(datasetindex);
			dataset->relloc();
			printPoints(dataset);
		}
		else
		{
			std::vector<int> datasetindex(numdatasets + 1, 0);
			int targetindex = 0;
			dataset = new DataSet(featureinfos, targetfeatureinfo, datasetname, n, 0, numdatasets);
			dataset->relloc();
			auto maxinstanceoffeature = dataset->getInstancesOfFeature();
			if (numdatasets > 1)dataset->serialize();
			csv2::Reader<delimiter<','>,
				quote_character<'"'>,
				first_row_is_header<false>,
				trim_policy::trim_whitespace> csvnohead;
			for (int datai = 1; datai < numdatasets; ++datai)
			{
				targetindex += dataset->getInstancesOfFeature();
				datasetindex[datai] = targetindex;
				datapath = datapathtemp + datapathfiles[datai];
				featureinfos = dataset->getFeatures();
				targetfeatureinfo = dataset->getTargetFeature();
				targetfeature = targetfeatureinfo->getFeature();
				if (csvnohead.mmap(datapath)) {
					//const auto header = csvnohead.header();
					int i = 0;
					n = csvnohead.rows();
					m = csvnohead.cols();

					i = 0;
					bool headflag = false;
					if (Property::getProperty()->getOtherDatasetHasHead())
						headflag = true;
					for (const auto& row : csvnohead) {
						int j = 0, featureindex = 0;
						if (headflag) { headflag = false; n--; continue; }
						if (n > maxinstanceoffeature)
						{
							LOG(ERROR) << "Load Dataset Error: Please use the most rows of dataset as the first dataset!";
							exit(-1);
						}
						for (const auto& cell : row) {
							// Do something with cell value
							std::string value;
							cell.read_value(value);
							if (j == targetcolindex)
							{

								if (classes >= 2)
								{
									int* val = new int;
									*val = std::stoi(value);
									setdiscrete.insert(*val);
									targetfeature->setValue(i + targetindex, reinterpret_cast<void*>(val));
									delete val;
								}
								else
								{
									MyDataType* val2 = new MyDataType;
									try {
										if (sizeof(MyDataType) == sizeof(float))
											*val2 = stof(value);
										else if (sizeof(MyDataType) == sizeof(double))
											*val2 = stod(value);
									}
									catch (...) {
										*val2 = 0.0;
									}
									targetfeature->setValue(i + targetindex, reinterpret_cast<void*>(val2));
									delete val2;
								}

								featureindex--;
							}
							else
							{
								auto feature = featureinfos[featureindex]->getFeature();
								if (feature->getType() == FeatureType::Discrete || feature->getType() == FeatureType::String)
								{
									int* val = new int;
									*val = std::stoi(value);
									feature->setValue(i, reinterpret_cast<void*>(val));
									delete val;
								}
								else if (feature->getType() == FeatureType::Date)
								{
									Date* val = new Date;
									*val = Date(value);
									feature->setValue(i, reinterpret_cast<void*>(val));
									delete val;
								}
								else
								{
									MyDataType* val = new MyDataType;
									try {
										if (sizeof(MyDataType) == sizeof(float))
											*val = stof(value);
										else if (sizeof(MyDataType) == sizeof(double))
											*val = stod(value);
									}
									catch (...) {
										if (sizeof(MyDataType) == sizeof(float))
											*val = NAN;
										else if (sizeof(MyDataType) == sizeof(double))
											*val = static_cast<double>(NAN);
									}
									feature->setValue(i, reinterpret_cast<void*>(val));
									delete val;
								}
							}
							j++;
							featureindex++;
						}
						i++;
					}
				}
				dataset->setID(datai);
				dataset->setInstancesOfFeature(n);
				dataset->serialize();
			}
			
			datasetindex[numdatasets] = targetindex + dataset->getInstancesOfFeature();
			dataset->setDatasetIndex(datasetindex);
			printPoints(dataset);
		}
		if(setdiscrete.size() != classes)
		{
			LOG(INFO) << "TargetClasses Num is " << setdiscrete.size() << " , but the default is " << classes << "!";
			if(ischange && !testdata) Property::getProperty()->setTargetClasses(setdiscrete.size());
		}
		auto vtargetvalue = targetfeature->getValues();
		if(classes >= 2)
		{
			std::unordered_map<int, int> mapdiscrete;
			int discretevalue = 0;
			for (const auto& val : setdiscrete)
				mapdiscrete[val] = discretevalue++;
			int* targetvalue = reinterpret_cast<int*>(vtargetvalue);
			std::transform(targetvalue, targetvalue + targetn, targetvalue, [&mapdiscrete](int val) { return mapdiscrete[val]; });
			auto targetfeatureinfo = dataset->getTargetFeature();
			auto feature = targetfeatureinfo->getFeature();
			feature->setNumsOfValues(discretevalue);
			targetfeatureinfo->setNumsOfValues(discretevalue);
		}
		return dataset;
	}
}//RL4FS
