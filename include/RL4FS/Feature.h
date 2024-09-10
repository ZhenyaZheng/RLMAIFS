#pragma once
#include "RL4FSStruct.h"
#include "Date.h"
#include "util/Property.h"
namespace RL4FS {
	class Feature
	{
	public:
		Feature() :m_size(0) {};
		virtual ~Feature() {};
		void virtualfunc() { ; }
		virtual void* getValue(int index) = 0;
		virtual void setValue(int index, void* val) = 0;
		virtual int getInstances() = 0;
		virtual void copyFeature(Feature* feature) = 0;
		virtual FeatureType getType() = 0;
		virtual void* getValues() = 0;
		virtual int getNumsOfValues() = 0;
		virtual void setNumsOfValues(int possiblevalues) = 0;
		virtual ostream& print(ostream& out)const = 0;
		friend ostream& operator<< (ostream& out, const Feature& feature);
		virtual void merge(Feature* feature, int length) = 0;
	protected:
		int m_size;
	};

	class DiscreteFeature :public Feature
	{
	private:
		int m_possiblevalues;
		std::vector<int> m_values;
	public:
		DiscreteFeature(int size, int possiblevalues = 1);
		~DiscreteFeature();
		int getNumsOfValues();
		void copyFeature(Feature* feature);
		FeatureType getType();
		ostream& print(ostream& out)const;
		void* getValue(int index);
		void setValue(int index, void* val);
		void* getValues();
		void setNumsOfValues(int possiblevalues);
		void merge(Feature* feature, int length);
		int getInstances();
	};


	class DateFeature : public Feature
	{
	public:
		DateFeature(int size);
		~DateFeature();
		FeatureType getType();
		ostream& print(ostream& out)const;
		int getNumsOfValues();
		void copyFeature(Feature* feature);
		void* getValue(int index);
		void setValue(int index, void* val);
		void* getValues();
		void setNumsOfValues(int possiblevalues);
		void merge(Feature* feature, int length);
		std::map<Date, std::vector<int>>* getDateKey();
		int getInstances();
	private:
		std::vector<Date> m_values;
		std::map<Date, std::vector<int>> *m_datekey;
	};

	class NumericFeature : public Feature
	{
	public:
		NumericFeature(int size);
		~NumericFeature();
		ostream& print(ostream& out)const;
		int getNumsOfValues();
		void copyFeature(Feature* feature);
		FeatureType getType();
		void* getValue(int index);
		void setValue(int index, void* val);
		void setNumsOfValues(int possiblevalues);
		void* getValues();
		void merge(Feature* feature, int length);
		int getInstances();
	private:
		std::vector<MyDataType> m_values;
	};

	class StringFeature : public Feature
	{
	public:
		StringFeature(int size);
		~StringFeature();
		void copyFeature(Feature* feature);
		ostream& print(ostream& out)const;
		int getNumsOfValues();
		FeatureType getType();
		void* getValue(int index);
		void setValue(int index, void* val);
		void setNumsOfValues(int possiblevalues);
		void* getValues();
		void merge(Feature* feature, int length);
		int getInstances();
	private:
		std::vector<string> m_values;
	};
}//RL4FS