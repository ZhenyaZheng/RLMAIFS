#include "RL4FS/Feature.h"
namespace RL4FS{
	ostream& operator<< (ostream& out, const Feature& feature)
	{
		return feature.print(out);
	}

	DiscreteFeature::DiscreteFeature(int size, int possiblevalues)
	{
		m_size = size;
		m_values.resize(size, 0);
		fill(m_values.begin(), m_values.end(), 0);
		m_possiblevalues = possiblevalues;
	}
	DiscreteFeature::~DiscreteFeature()
	{
		m_values.clear();
	}
	int DiscreteFeature::getNumsOfValues()
	{
		return m_possiblevalues;
	}
	void DiscreteFeature::copyFeature(Feature* feature)
	{
		auto n = feature->getInstances();
		//for (int i = 0; i < n; ++i) *(m_values.begin() + i) = *(reinterpret_cast<int*> (feature->getValues()) + i);
		// quick copy
		memcpy(m_values.data(), feature->getValues(), n * sizeof(int));
	}
	FeatureType DiscreteFeature::getType()
	{
		return FeatureType::Discrete;
	}
	ostream& DiscreteFeature::print(ostream& out)const
	{
		for (int i = 0; i < m_size; ++i)
			out << *(m_values.begin() + i) << " ";
		out << endl;
		return out;
	}
	void* DiscreteFeature::getValue(int index) { return index < m_size ? reinterpret_cast<void*>(&(m_values[index])) : nullptr; }
	void DiscreteFeature::setValue(int index, void* val)
	{
		if (index < m_size)
			*(m_values.begin() + index) = *(reinterpret_cast<int*> (val));

	}
	void* DiscreteFeature::getValues()
	{
		return getValue(0);
	}
	void DiscreteFeature::setNumsOfValues(int possiblevalues)
	{
		m_possiblevalues = possiblevalues;
	}
	void DiscreteFeature::merge(Feature* feature, int length)
	{
		m_values.insert(m_values.end(), reinterpret_cast<int*>(feature->getValues()), reinterpret_cast<int*>(feature->getValues()) + length);
		m_possiblevalues = std::max(m_possiblevalues, reinterpret_cast<DiscreteFeature*>(feature)->m_possiblevalues);
		m_size += length;
	}
	int DiscreteFeature::getInstances()
	{
		return m_size;
	}




	DateFeature::DateFeature(int size) { m_size = size; m_values.resize(size); m_datekey = new std::map<Date, std::vector<int>>; }
	DateFeature::~DateFeature() { m_values.clear(); m_datekey->clear(); delete m_datekey; m_datekey = nullptr; }
	FeatureType DateFeature::getType()
	{
		return FeatureType::Date;
	}
	ostream& DateFeature::print(ostream& out)const
	{
		for (int i = 0; i < m_size; ++i)
			out << *(m_values.begin() + i) << " ";
		out << endl;
		return out;
	}
	int DateFeature::getNumsOfValues()
	{
		return -1;
	}
	void DateFeature::setNumsOfValues(int possiblevalues)
	{
		return;
	}
	void DateFeature::copyFeature(Feature* feature)
	{
		auto n = feature->getInstances();
		//for (int i = 0; i < n; ++i) *(m_values.begin() + i) = *(reinterpret_cast<Date*> (feature->getValues()) + i);
		// quick copy
		memcpy(m_values.data(), feature->getValues(), n * sizeof(Date));

	}
	void* DateFeature::getValue(int index) { return index < m_size ? reinterpret_cast<void*>(&(m_values[index])) : nullptr;}
	void DateFeature::setValue(int index, void* val)
	{
		Date date = *(reinterpret_cast<Date*> (val));
		if (index < m_size)
			*(m_values.begin() + index) = date;
		if ((*m_datekey).find(date) != (*m_datekey).end())
			(*m_datekey)[date].push_back(index);
		else (*m_datekey).insert({ date, std::vector<int>({index}) });

	}
	void* DateFeature::getValues()
	{
		return getValue(0);
	}
	std::map<Date, std::vector<int>>* DateFeature::getDateKey()
	{
		return m_datekey;
	}
	void DateFeature::merge(Feature* feature, int length)
	{
		m_values.insert(m_values.end(), reinterpret_cast<Date*>(feature->getValues()), reinterpret_cast<Date*>(feature->getValues()) + length);
		m_size += length;
		auto datekey = reinterpret_cast<DateFeature*>(feature)->getDateKey();
		for (auto it = datekey->begin(); it != datekey->end(); ++it)
		{
			if ((*m_datekey).find(it->first) != (*m_datekey).end())
				(*m_datekey)[it->first].insert((*m_datekey)[it->first].end(), it->second.begin(), it->second.end());
			else (*m_datekey).insert({ it->first, it->second });
		}
	}
	int DateFeature::getInstances()
	{
		return m_size;
	}


	NumericFeature::NumericFeature(int size) { m_size = size; m_values.resize(size, 0); }
	NumericFeature::~NumericFeature()
	{
		m_values.clear();
	}
	ostream& NumericFeature::print(ostream& out)const
	{
		for (int i = 0; i < m_size; ++i)
			out << *(m_values.begin() + i) << " ";
		out << endl;
		return out;
	}
	int NumericFeature::getNumsOfValues()
	{
		return -1;
	}
	void NumericFeature::setNumsOfValues(int possiblevalues)
	{
		return;
	}
	void NumericFeature::copyFeature(Feature* feature)
	{
		auto n = feature->getInstances();
		//for (int i = 0; i < n; ++i) *(m_values.begin() + i) = *(reinterpret_cast<MyDataType*> (feature->getValues()) + i);
		// quick copy
		memcpy(m_values.data(), feature->getValues(), n * sizeof(MyDataType));
	}
	FeatureType NumericFeature::getType()
	{
		return FeatureType::Numeric;
	}
	void* NumericFeature::getValue(int index) { return index < m_size ? reinterpret_cast<void*>(&(m_values[index])): nullptr;}
	void NumericFeature::setValue(int index, void* val)
	{
		if (index < m_size)
			*(m_values.begin() + index) = *(reinterpret_cast<MyDataType*> (val));

	}
	void* NumericFeature::getValues()
	{
		return getValue(0);
	}
	void NumericFeature::merge(Feature* feature, int length)
	{
		m_values.insert(m_values.end(), reinterpret_cast<MyDataType*>(feature->getValues()), reinterpret_cast<MyDataType*>(feature->getValues()) + length);
		m_size += length;
	}
	int NumericFeature::getInstances()
	{
		return m_size;
	}





	StringFeature::StringFeature(int size) { m_size = size; m_values.resize(size); }
	StringFeature::~StringFeature()
	{

		m_values.clear();
	}
	void StringFeature::copyFeature(Feature* feature)
	{
		auto n = feature->getInstances();
		for (int i = 0; i < n; ++i)
			*(m_values.begin() + i) = *(reinterpret_cast<string*> (feature->getValues()) + i);
	}
	ostream& StringFeature::print(ostream& out)const
	{
		for (int i = 0; i < m_size; ++i)
			out << *(m_values.begin() + i) << " ";
		out << endl;
		return out;
	}
	int StringFeature::getNumsOfValues()
	{
		return -1;
	}
	FeatureType StringFeature::getType()
	{
		return FeatureType::String;
	}
	void* StringFeature::getValue(int index) { return index < m_size ? reinterpret_cast<void*>(&(m_values[index])) : nullptr;}
	void StringFeature::setValue(int index, void* val)
	{
		if (index < m_size)
			*(m_values.begin() + index) = *(reinterpret_cast<string*> (val));

	}
	void StringFeature::setNumsOfValues(int possiblevalues)
	{
		return;
	}
	void* StringFeature::getValues()
	{
		return getValue(0);
	}

	void StringFeature::merge(Feature* feature, int length)
	{
		m_values.insert(m_values.end(), reinterpret_cast<string*>(feature->getValues()), reinterpret_cast<string*>(feature->getValues()) + length);
		m_size += length;
	}

	int StringFeature::getInstances()
	{
		return m_size;
	}

}//RL4FS