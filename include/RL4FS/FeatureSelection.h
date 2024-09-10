#pragma once
#include "DataSet.h"
using std::pair;
namespace RL4FS
{
	class FeatureSelection
	{
		public:
			FeatureSelection();
			~FeatureSelection();
			void run(DataSet* dataset, vector<int>& selectionvec);

			MyDataType produceScore(DataSet* dataset, PFeatureInfo& featureinfo);

			MyDataType produceScoreIG(DataSet* dataset, PFeatureInfo& featureinfo);

			MyDataType produceScoreANOVA(DataSet* dataset, PFeatureInfo &featureinfo);

			MyDataType calculateIG(int instances, std::unordered_map<std::string, std::vector<int>>& valuesperkey);

			MyDataType calculateANOVA(const std::unordered_map<std::string, std::vector<MyDataType>>& groups);

			MyDataType getSteps(MyDataType* value, int instances, MyDataType &minvalue, MyDataType &maxvalue, int bins);
	};

}