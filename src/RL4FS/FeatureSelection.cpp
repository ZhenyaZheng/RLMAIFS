#include "RL4FS/FeatureSelection.h"

namespace RL4FS {
	FeatureSelection::FeatureSelection()
	{
	}

	FeatureSelection::~FeatureSelection()
	{
	}

	void FeatureSelection::run(DataSet* dataset, vector<int>& selectionvec)
	{
		int featurenum = dataset->getFeatureSize(false);
		vector<pair<MyDataType, int>> scorevec(featurenum, {0, 0});
		int threadsnum = Property::getProperty()->getWeThreadNum();
		for (int dataid = 0; dataid < dataset->getNumID(); dataid++)
		{
			dataset->setID(dataid);
			dataset->deserialize();
			#pragma omp parallel for num_threads(threadsnum)
			for (int i = 0; i < featurenum; i++)
			{
				PFeatureInfo featureinfo = dataset->getFeature(i, true);
				MyDataType score = produceScore(dataset, featureinfo);
				scorevec[i].first += score;
				scorevec[i].second = i;
				dataset->clearFeatureData(featureinfo);
			}
		}
		if (Property::getProperty()->getDistributedNodes() > 1)
		{
#ifdef USE_MPICH
			int process_id;
			MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
			int process_num;
			MPI_Comm_size(MPI_COMM_WORLD, &process_num);
			vector<MyDataType> scorevecall(featurenum);
			for (int i = 0; i < featurenum; i++)
				scorevecall[i] = scorevec[i].first;
			vector<MyDataType> scorevecallsum(featurenum);
			if (sizeof (MyDataType) == sizeof (float))
				MPI_Allreduce(scorevecall.data(), scorevecallsum.data(), featurenum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
			else
				MPI_Allreduce(scorevecall.data(), scorevecallsum.data(), featurenum, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			for (int i = 0; i < featurenum; i++)
				scorevec[i].first = scorevecallsum[i];
#else
			LOG(FATAL) << "filterFeatureByIG Please USE_MPICH = ON when CMake this project, or you cannot set Property::getProperty()->setDistributedNodes() > 1";
#endif // 
		}
		sort(scorevec.begin(), scorevec.end(), [](const pair<MyDataType, int>& a, const pair<MyDataType, int>& b) {return a.first > b.first; });
		int maxnumsfeatures = Property::getProperty()->getMaxNumsFeatures();
		if (maxnumsfeatures > 0 && maxnumsfeatures < featurenum)
		{
			for (int i = 0; i < maxnumsfeatures; i++)
				selectionvec.push_back(scorevec[i].second);
		}
		else
		{
			for (int i = 0; i < featurenum; i++)
				selectionvec.push_back(scorevec[i].second);
			selectionvec.resize(max(featurenum / 4, 1));
		}
	}

	MyDataType FeatureSelection::produceScore(DataSet* dataset, PFeatureInfo& featureinfo)
	{
		auto targetfeature = dataset->getTargetFeature()->getFeature();
		if (Property::getProperty()->getTargetClasses() <= 0 || targetfeature->getType() == FeatureType::Numeric) return -produceScoreANOVA(dataset, featureinfo);
		return produceScoreIG(dataset, featureinfo);
	}

	MyDataType FeatureSelection::getSteps(MyDataType* value, int instances, MyDataType& minvalue, MyDataType& maxvalue, int bins)
	{
		for (int i = 0; i < instances; i++)
		{
			if (value[i] < minvalue) minvalue = value[i];
			if (value[i] > maxvalue) maxvalue = value[i];
		}
		return (maxvalue - minvalue) / bins;
	}

	MyDataType FeatureSelection::produceScoreIG(DataSet* dataset, PFeatureInfo& featureinfo)
	{
		try{
			int targetindex = dataset->getDatasetIndex()[dataset->getID()];
			auto targetfeature = dataset->getTargetFeature()->getFeature();
			auto instances = dataset->getInstancesOfFeature();
			if (dataset->getNumID() > 1) instances = dataset->getInstancesOfFeature(3);
			std::unordered_map<std::string, std::vector<int>> valuesperkey;
			MyDataType minvalue = 1e9, maxvalue = -1e9;
			int bins = max(min(255, instances/10), 1);
			MyDataType steps = 1.0;

			if(featureinfo->getType() == OutType::Numeric) steps = getSteps(reinterpret_cast<MyDataType*>(featureinfo->getFeature()->getValue(0)), instances, minvalue, maxvalue, bins);
			for (int i = 0; i < instances; ++i)
			{
				string key;
				if (featureinfo->getType() == OutType::Numeric)
				{
					int index = (reinterpret_cast<MyDataType*>(featureinfo->getFeature()->getValue(i))[0] - minvalue) / steps;
					key += std::to_string(index);
				}
				else key += std::to_string(*(reinterpret_cast<int*>(featureinfo->getFeature()->getValue(i))));
				auto label = *reinterpret_cast<int*> (targetfeature->getValue(i + targetindex));
				std::vector<int> labelvalue(targetfeature->getNumsOfValues(), 0);
				if (valuesperkey[key].size() == 0)
					valuesperkey[key] = labelvalue;
				valuesperkey[key][label]++;
			}
			auto score = calculateIG(instances, valuesperkey);
			return score;
		}
		catch (...)
		{
			LOG(ERROR) << "FeatureSelection produceScore Error!";
			return 0.0;
		}
	}

	MyDataType FeatureSelection::produceScoreANOVA(DataSet* dataset, PFeatureInfo& featureinfo)
	{
		try
		{
			auto targetfeature = dataset->getTargetFeature()->getFeature();
			if (targetfeature->getType() != FeatureType::Numeric) {
				LOG(ERROR) << "Target feature is not numeric!";
				return -1e9;
			}
			auto instances = dataset->getInstancesOfFeature();
			int targetindex = dataset->getDatasetIndex()[dataset->getID()];

			if (dataset->getNumID() > 1) instances = dataset->getInstancesOfFeature(3);
			std::unordered_map<std::string, std::vector<MyDataType>> groups;
			MyDataType minvalue = 1e9, maxvalue = -1e9;
			int bins = max(min(255, instances / 10), 1);
			MyDataType steps = 1.0;
			if (featureinfo->getType() == OutType::Numeric) steps = getSteps(reinterpret_cast<MyDataType*>(featureinfo->getFeature()->getValue(0)), instances, minvalue, maxvalue, bins);
			for (int i = 0; i < instances; ++i)
			{
				std::string key;
				if (featureinfo->getType() == OutType::Numeric)
				{
					int index = (reinterpret_cast<MyDataType*>(featureinfo->getFeature()->getValue(i))[0] - minvalue) / steps;
					key += std::to_string(index);
				}
				else key += std::to_string(*(reinterpret_cast<int*>(featureinfo->getFeature()->getValue(i))));
				auto label = *reinterpret_cast<MyDataType*> (targetfeature->getValue(i + targetindex));
				groups[key].push_back(label);
			}
			auto score = calculateANOVA(groups);
			return score;
		}
		catch (...)
		{
			LOG(ERROR) << "FeatureSelection produceScore Error!";
			return -1e9;
		}
	}

	MyDataType FeatureSelection::calculateIG(int instances, std::unordered_map<std::string, std::vector<int>>& valuesperkey)
	{
		MyDataType ig = 0;
		for (auto& val : valuesperkey)
		{
			auto numofinstance = std::accumulate(val.second.begin(), val.second.end(), 0.0);
			if (numofinstance < 1.0)continue;
			MyDataType tempig = 0.0;
			for (auto& va : val.second)
				if (va > 0)
				{
					tempig += -((va / (MyDataType)numofinstance) * log2(va / (MyDataType)numofinstance));
				}
			ig += ((MyDataType)numofinstance / instances) * tempig;
		}
		return ig;
	}

	MyDataType FeatureSelection::calculateANOVA(const std::unordered_map<std::string, std::vector<MyDataType>>& groups)
	{
		int n = 0; // total number of instances
		MyDataType ss_total = 0.0; // total sum of squares
		MyDataType ss_between = 0.0; // total sum of squares between groups
		MyDataType grand_mean = 0.0; // total mean
		std::unordered_map<std::string, MyDataType> groupmeans;
		long double sum = 0.0;
		for (const auto& pair : groups)
		{
			const auto& values = pair.second;
			MyDataType groupsum = std::accumulate(values.begin(), values.end(), 0.0);
			sum += groupsum;
			groupmeans[pair.first] = groupsum / values.size();
			n += values.size();
		}
		grand_mean = sum / n;
		for (const auto& pair : groups)
		{
			const auto& values = pair.second;
			MyDataType group_mean = groupmeans[pair.first];
			ss_between += values.size() * std::pow(group_mean - grand_mean, 2);
			for (MyDataType value : values)
			{
				ss_total += std::pow(value - grand_mean, 2);
			}
		}
		MyDataType ss_within = ss_total - ss_between; // the sum of squares within groups
		int df_between = groups.size() - 1; // the between-group degrees of freedom
		int df_within = n - groups.size(); // the within-group degrees of freedom
		MyDataType ms_between = ss_between / df_between; // the between-group mean square
		MyDataType ms_within = ss_within / df_within; // the within-group mean square
		MyDataType f = ms_between / ms_within; // the F-value
		//MyDataType p = 1 - MyMath<MyDataType>::f_cdf(f, df_between, df_within); // the p-value
		return f;
	}
}