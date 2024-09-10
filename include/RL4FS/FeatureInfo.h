#pragma once
#include "./Feature.h"
#include <utility>
namespace RL4FS {

	class FeatureInfo
	{
		using PFeatureInfo = FeatureInfo*;
	private:
		bool m_issecondfeature = false;
		int m_possiblevalues = -1;
		int m_index = -1;
		OutType m_type;
		Feature* m_feature;
		std::vector<PFeatureInfo> m_sourcefeatures;
		std::vector<PFeatureInfo> m_targetfeatures;
		string m_name;
	public:
		FeatureInfo();
		FeatureInfo(Feature* feature, std::vector<PFeatureInfo> sourcefeatures, std::vector<PFeatureInfo> targetfeatures, OutType type, string name, bool issecondfeature=false, int possiblevalues=-1);
		~FeatureInfo() {};
		void setFeature(Feature* feature) { m_feature = feature; }
		PFeatureInfo generateSubFeatureInfo(const std::vector<int> &index);
		bool isEmpty() const{ return m_feature == nullptr; };
		FeatureInfo(const FeatureInfo&);
		FeatureInfo(FeatureInfo&&);
		FeatureInfo operator= (FeatureInfo&&);
		FeatureInfo operator= (const FeatureInfo&);
		bool in(const std::vector<PFeatureInfo>&)const;
		bool clear(bool isclearsource=true);
		void setSourceFeatures(const std::vector<PFeatureInfo>&);
		void setTargetFeatures(const std::vector<PFeatureInfo>&);
		Feature* getFeature()const;
		std::vector<PFeatureInfo> getSourceFeatures()const;
		std::vector<PFeatureInfo> getTargetFeatures()const;
		OutType  getType()const;
		bool getIsSecondFeature()const;
		void setIsSecondFeature(bool issecondfeature);
		string getName()const;
		bool operator==(const PFeatureInfo&)const;
		bool operator!=(const PFeatureInfo&)const;
		PFeatureInfo copy(const PFeatureInfo& featureinfo);
		int getNumsOfValues()const;
		void setNumsOfValues(int possiblevalues);
		int getIndex()const;
		void setIndex(int index);
		void setName(const string& name);
	};
	using PFeatureInfo = FeatureInfo*;
}//RL4FS

