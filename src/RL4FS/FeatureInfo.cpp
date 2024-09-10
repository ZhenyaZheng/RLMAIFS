#include "RL4FS/FeatureInfo.h"
namespace RL4FS {
    FeatureInfo::FeatureInfo() :m_feature(nullptr), m_issecondfeature(false), m_type(OutType::Numeric)
    {
    }

    FeatureInfo::FeatureInfo(Feature* feature, std::vector<PFeatureInfo> sourcefeatures, std::vector<PFeatureInfo> targetfeatures, OutType type, string name, bool issecondfeature, int possiblevalues)
    {
        m_feature = feature;
        m_sourcefeatures = sourcefeatures;
        m_targetfeatures = targetfeatures;
        m_type = type;
        m_issecondfeature = issecondfeature;
        m_name = name;
        m_possiblevalues = possiblevalues;
    }
    
    FeatureInfo::FeatureInfo(const FeatureInfo& featureinfo)
    {
        m_feature = featureinfo.getFeature();
        m_sourcefeatures = featureinfo.getSourceFeatures();
        m_targetfeatures = featureinfo.getTargetFeatures();
        m_issecondfeature = featureinfo.getIsSecondFeature();
        m_name = string(featureinfo.getName());
        m_type = featureinfo.getType();
        m_possiblevalues = featureinfo.getNumsOfValues();
        m_index = featureinfo.getIndex();
    }

    FeatureInfo FeatureInfo::operator=(const FeatureInfo& featureinfo)
    {
        m_feature = featureinfo.getFeature();
        m_sourcefeatures = featureinfo.getSourceFeatures();
        m_targetfeatures = featureinfo.getTargetFeatures();
        m_issecondfeature = featureinfo.getIsSecondFeature();
        m_name = string(featureinfo.getName());
        m_type = featureinfo.getType();
        m_possiblevalues = featureinfo.getNumsOfValues();
        m_index = featureinfo.getIndex();
        return *this;
    }

    FeatureInfo::FeatureInfo(FeatureInfo&& featureinfo)
    {
        m_feature = featureinfo.getFeature();
        m_sourcefeatures = featureinfo.getSourceFeatures();
        m_targetfeatures = featureinfo.getTargetFeatures();
        m_issecondfeature = featureinfo.getIsSecondFeature();
        m_name = string(featureinfo.getName());
        m_type = featureinfo.getType();
        m_possiblevalues = featureinfo.getNumsOfValues();
        m_index = featureinfo.getIndex();
    }

    FeatureInfo FeatureInfo::operator= (FeatureInfo&& featureinfo)
    {
        m_feature = featureinfo.getFeature();
        m_sourcefeatures = featureinfo.getSourceFeatures();
        m_targetfeatures = featureinfo.getTargetFeatures();
        m_issecondfeature = featureinfo.getIsSecondFeature();
        m_name = string(featureinfo.getName());
        m_type = featureinfo.getType();
        m_possiblevalues = featureinfo.getNumsOfValues();
        m_index = featureinfo.getIndex();
        return *this;
    }

    PFeatureInfo FeatureInfo::copy(const PFeatureInfo& featureinfo)
    {
        auto oldfeature = featureinfo->getFeature();
        if (oldfeature == nullptr)
        {
            m_feature = nullptr;
        }
        else
        {
            int n = oldfeature->getInstances();
            FeatureType type = oldfeature->getType();
            //void* feature_ptr = nullptr;

            switch (type)
            {
            case FeatureType::Date: m_feature = new DateFeature(n); break;
            case FeatureType::Discrete: m_feature = new DiscreteFeature(n, oldfeature->getNumsOfValues()); break;
            case FeatureType::Numeric: m_feature = new NumericFeature(n); break;
            case FeatureType::String: m_feature = new StringFeature(n); break;
            }
            m_feature->copyFeature(featureinfo->getFeature());
        }
        
        if (featureinfo->getIsSecondFeature())
        {
            PFeatureInfo newsourcefeatureinfo = new FeatureInfo();
            newsourcefeatureinfo->copy(featureinfo->getSourceFeatures()[0]);
            m_sourcefeatures.push_back(newsourcefeatureinfo);
        }
        else m_sourcefeatures = featureinfo->getSourceFeatures();
        m_targetfeatures = featureinfo->getTargetFeatures();
        m_type = featureinfo->getType();
        m_issecondfeature = featureinfo->getIsSecondFeature();
        m_name = featureinfo->getName();
        return this;
    }

    PFeatureInfo FeatureInfo::generateSubFeatureInfo(const std::vector<int>& index)
    {
        Feature* newfeature = nullptr;
        const int n = index.size();
        FeatureType type = m_feature->getType();
        switch (type)
        {
        case FeatureType::Date: newfeature = new DateFeature(n); break;
        case FeatureType::Discrete: newfeature = new DiscreteFeature(n); break;
        case FeatureType::Numeric: newfeature = new NumericFeature(n); break;
        case FeatureType::String: newfeature = new StringFeature(n); break;
        }
        for (int i = 0; i < n; ++i)
        {
            switch (type)
            {
                case FeatureType::Date: 
                {
                    Date* val = new Date();
                    *val = *reinterpret_cast<Date*>(m_feature->getValue(index[i]));
                    newfeature->setValue(i, reinterpret_cast<void*>(val));
                    delete val;
                    break;
                }
                case FeatureType::Discrete: 
                {
                    int* val = new int;
                    *val = *reinterpret_cast<int*>(m_feature->getValue(index[i]));
                    newfeature->setValue(i, reinterpret_cast<void*>(val));
                    delete val;
                    break;
                }
                case FeatureType::Numeric: 
                {
                    MyDataType* val = new MyDataType();
                    *val = *reinterpret_cast<MyDataType*>(m_feature->getValue(index[i]));
                    newfeature->setValue(i, reinterpret_cast<void*>(val));
                    delete val;
                    break;
                }
                case FeatureType::String: 
                {
                    string* val = new string();
                    *val = *reinterpret_cast<string*>(m_feature->getValue(index[i]));
                    newfeature->setValue(i, reinterpret_cast<void*>(val));
                    delete val;
                    break;
                }
            }
            
        }
        auto sourcefeature = m_sourcefeatures;
        auto targetfeature = m_targetfeatures;
        auto otype = m_type;
        auto issecondfeature = m_issecondfeature;
        auto name = m_name;
        return new FeatureInfo(newfeature, sourcefeature, targetfeature, otype, name, issecondfeature);
    }

    bool FeatureInfo::in(const std::vector<PFeatureInfo>& featureinfos)const
    {
        for (auto& feain : featureinfos)
        {
            if (*this == feain)
                return true;
        }
        return false;
    }
    

    bool FeatureInfo::clear(bool isclearsource)
    {
        if (m_issecondfeature && isclearsource && m_sourcefeatures.size() == 1 && m_sourcefeatures[0]->getFeature())
        {
            m_sourcefeatures[0]->clear();
            delete m_sourcefeatures[0];
        }
        if (m_feature) delete m_feature;
        m_feature = nullptr;
        m_sourcefeatures.clear();
        m_targetfeatures.clear();
        m_feature = nullptr;
        m_name.clear();
        return true;
    }

    Feature* FeatureInfo::getFeature()const
    {
        return m_feature;
    }
    std::vector<PFeatureInfo> FeatureInfo::getSourceFeatures()const
    {
        return m_sourcefeatures;
    }
    std::vector<PFeatureInfo> FeatureInfo::getTargetFeatures()const
    {
        return m_targetfeatures;
    }

    void FeatureInfo::setSourceFeatures(const std::vector<PFeatureInfo>& sourcefeature)
    {
        m_sourcefeatures = sourcefeature;
    }

    void FeatureInfo::setTargetFeatures(const std::vector<PFeatureInfo>& targetfeature)
    {
        m_targetfeatures = targetfeature;
    }

    OutType FeatureInfo::getType()const
    {
        return m_type;
    }
    bool FeatureInfo::getIsSecondFeature()const
    {
        return m_issecondfeature;
    }
    void FeatureInfo::setIsSecondFeature(bool issecondfeature)
    {
		m_issecondfeature = issecondfeature;
    }
    string FeatureInfo::getName()const
    {
        return m_name;
    }
    bool FeatureInfo::operator==(const PFeatureInfo& featureinfo)const
    {
        if (!featureinfo) return false;
        if (m_name == featureinfo->getName() && m_type == featureinfo->getType())
            return true;
        return false;
    }
    bool FeatureInfo::operator!=(const PFeatureInfo& featureinfo)const
    {
        return !(*this == featureinfo);
    }
    int FeatureInfo::getNumsOfValues()const
    {
		return m_possiblevalues;
	}
    void FeatureInfo::setNumsOfValues(int possiblevalues)
    {
		m_possiblevalues = possiblevalues;
	}
    int FeatureInfo::getIndex() const
    {
        return m_index;
    }
    void FeatureInfo::setIndex(int index)
	{
        m_index = index;
	}
	void FeatureInfo::setName(const string& name)
	{
		m_name = name;
	}
}//RL4FS