#include "RL4FS/rl/model.h"

namespace RL4FS {
    Model::Model(DataSet* dataset):m_fcdataset(dataset)
    {initdata();}
    Model::~Model()
    {}

    double Model::getInitLoss(){
        return m_init_loss;
    }

    double Model::getScore()
    {
        return m_score;
    }

    double Model::getLoss(const vector<unsigned int>& state)
    {
        std::unordered_set<int> remainfeatureindexs(m_featureindex.begin(), m_featureindex.end());
        CCHECK_EQ(state.size(), m_featureindexmap.size(), "Model getLoss state.size() != m_fearureindexmap.size()");
        vector<int> featureindexs;
        string statestr = stateToString(state);
        // lock the m_statescores
        {
            std::lock_guard<std::mutex> lock(m_statescores_lock);
            if (m_statescores.find(statestr) != m_statescores.end())
                return m_statescores[statestr];
        }
        for(int i = 0;i < state.size(); ++i) {
            if(state[i] == 0) continue;
            if (m_featureindexmap.find(i) != m_featureindexmap.end()) 
                featureindexs.push_back(m_featureindexmap[i]);
            else LOG (ERROR) << "The feature index is not in the feature index map";
        }
        if(featureindexs.size() == 0) return m_score;
        auto dataset = generateLightGBMDatasetFromMat(featureindexs);
        //subset
        vector<int> trainindexs(m_fcdataset->m_trainindex), valindexs(m_fcdataset->m_valindex);
        shuffle(trainindexs.begin(), trainindexs.end(), std::default_random_engine(0));
        shuffle(valindexs.begin(), valindexs.end(), std::default_random_engine(0));
        trainindexs.resize(std::min(std::max(std::min(static_cast<int>(trainindexs.size()), 10000),500), static_cast<int>(trainindexs.size())));
        valindexs.resize(std::min(std::max(std::min(static_cast<int>(valindexs.size()), 10000), 300), static_cast<int>(valindexs.size())));
        std::sort(trainindexs.begin(), trainindexs.end());
        std::sort(valindexs.begin(), valindexs.end());
        auto subtrain = getSubDataset(dataset, trainindexs);
        auto subvalid = getSubDataset(dataset, valindexs);
        LGBM_DatasetFree(dataset);
        auto booster = InitBooster(subtrain, subvalid);
        int is_finished = 0;
        int out_len = 0;
        LGBM_BoosterGetEvalCounts(booster, &out_len);
        double* out_results = new double[out_len];
        int earlystopcount = 0;
        double bestscore = std::numeric_limits<double>::max();
        double score = 0;
        int bestiter = 0;
        for(int iter = 0; iter < 100; ++iter)
        {
            int flag = LGBM_BoosterUpdateOneIter(booster, &is_finished);
            checkStatus(flag, "Model getLoss BoosterUpdateOneIter");
            if (subvalid != nullptr)
                LGBM_BoosterGetEval(booster, 1, &out_len, out_results);
            else
                LGBM_BoosterGetEval(booster, 0, &out_len, out_results);
            score = out_results[0];
            if (earlyStop(score, 20, earlystopcount, bestscore))break;
            if (abs(score - bestscore) < 1e-20)bestiter = iter + 1;
            if (is_finished) break;
        }
        delete[] out_results;
        LGBM_BoosterFree(booster);
        LGBM_DatasetFree(subtrain);
        LGBM_DatasetFree(subvalid);
        // update the m_statescores
        {
			std::lock_guard<std::mutex> lock(m_statescores_lock);
			m_statescores[statestr] = bestscore;
		}
        return bestscore;        
    }

    void Model::initdata()
    {
        clear();
        auto dataset = m_fcdataset;
        if(dataset == nullptr)
        {
            LOG(ERROR) << "The dataset is nullptr";
            exit(1);
        }
        vector<double> scores(dataset->getFeatureSize(false), 0);
        selectImportantFeature(scores);
        vector<int> featureindexs;
        for (int i = 0; i < dataset->getFeatureSize(false); ++i)
            featureindexs.push_back(i);
        // select the feature
        int selectnum = Property::getProperty()->getMaxNumsFeatures();
        if(selectnum <= 0)selectnum = std::max(dataset->getFeatureSize(false) / 2, 1);
        vector<std::pair<double, int>> scoreindexs;
        for (int i = 0; i < scores.size(); ++i)
            scoreindexs.push_back(std::make_pair(scores[i], i));
        std::sort(scoreindexs.begin(), scoreindexs.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {return a.first > b.first; });
        m_featureindex.clear();
        for (int i = 0; i < selectnum; ++i)
            m_featureindex.push_back(scoreindexs[i].second);
        std::unordered_set<int> remainfeatureindexs(m_featureindex.begin(), m_featureindex.end());
        for(int i = 0;i < m_fcdataset->getFeatureSize(false); ++i)
        {
            if(remainfeatureindexs.find(i) == remainfeatureindexs.end())
            {
                int index = m_featureindexmap.size();
                m_featureindexmap.insert(std::make_pair(index, i));
            }
        }
        CCHECK_EQ(m_featureindex.size() + m_featureindexmap.size(), m_fcdataset->getFeatureSize(false), "Model initdata m_featureindex.size() + m_featureindexmap.size() != m_fcdataset->getFeatureSize(false)");
        LOG(INFO) << "Start to init the dataset for the model";
        DatasetHandle data = generateLightGBMDatasetFromMat(m_featureindex);
        int instancesoftrain = m_fcdataset->m_trainindex.size();
        int instancesofvalid = m_fcdataset->m_valindex.size();
        auto train_data = getSubDataset(data, m_fcdataset->m_trainindex);
        auto valid_data = getSubDataset(data, m_fcdataset->m_valindex);
        LGBM_DatasetFree(data);
        auto booster = InitBooster(train_data, valid_data);
        int is_finished = 0;
        int out_len = 0;
        LGBM_BoosterGetEvalCounts(booster, &out_len);

        double * out_results = new double[out_len];
        int earlystopcount = 0;
        const auto &lightgbmparams = Property::getProperty()->getLightGBMParams();
        int iterations = lightgbmparams.find("num_iterations") == lightgbmparams.end() ? 100 : std::stoi(lightgbmparams.find("num_iterations")->second);
        int earlystopround = lightgbmparams.find("early_stopping_round") == lightgbmparams.end() ? 10 : std::stoi(lightgbmparams.find("early_stopping_round")->second);
        double bestscore = (std::numeric_limits<double>::max)();
        double score = 0;
        int numclass = Property::getProperty()->getTargetClasses();
        for(int iter =0;iter < 10000;++ iter)
        {
            int flag = LGBM_BoosterUpdateOneIter(booster, &is_finished);
            checkStatus(flag, "Model InitDataSet BoosterUpdateOneIter");
            LGBM_BoosterGetEval(booster, 1, &out_len, out_results);
            score = out_results[0];
            if(earlyStop(score, 200, earlystopcount, bestscore)) break;
            if (is_finished) break;
        }
        m_score = bestscore;
        delete[] out_results;
        int numinstances = m_fcdataset->getInstancesOfFeature();
        numclass = numclass <= 2 ? 1 : numclass;
        out_results = new double[numclass * numinstances];
        int64_t out_len2 = 0, out_len3 = 0;
        LGBM_BoosterGetPredict(booster, 0, &out_len2, out_results);
        m_init_scores.resize(numclass * numinstances);
        for (int i = 0;i < m_fcdataset->m_trainindex.size();++ i)
        {
            int iindex = m_fcdataset->m_trainindex[i];
            for (int j = 0; j < numclass; ++j)
            {
                int index = j * numinstances + iindex;
                int oindex = j * instancesoftrain + i;
				m_init_scores[index] = out_results[oindex];
			}
        }
        LGBM_BoosterGetPredict(booster, 1, &out_len3, out_results);
        for (int i = 0; i < m_fcdataset->m_valindex.size(); ++i)
        {
            int iindex = m_fcdataset->m_valindex[i];
            for (int j = 0; j < numclass; ++j)
            {
                int index = j * numinstances + iindex;
                int oindex = j * instancesofvalid + i;
                m_init_scores[index] = out_results[oindex];
            }
        }
        delete [] out_results;
        LGBM_BoosterFree(booster);
        LGBM_DatasetFree(train_data);
        LGBM_DatasetFree(valid_data);
        booster = nullptr;
        LOG(INFO) << "The init score of the model: " << bestscore;
    }

    void Model::selectImportantFeature(vector<double> &scores)
    {
        auto dataset = generateLightGBMDatasetFromMat();
        auto subtrain = getSubDataset(dataset, m_fcdataset->m_trainindex);
        auto subvalid = getSubDataset(dataset, m_fcdataset->m_valindex);
        LGBM_DatasetFree(dataset);
        auto booster = InitBooster(subtrain, subvalid);
        int is_finished = 0;
        int out_len = 0;
        LGBM_BoosterGetEvalCounts(booster, &out_len);
        double* out_results = new double[out_len];
        int earlystopcount = 0;
        double bestscore = 1e10;
        double score = 0;
        int bestiter = 0;
        for(int iter =0;iter < 10000;++ iter)
        {
            int flag = LGBM_BoosterUpdateOneIter(booster, &is_finished);
            checkStatus(flag, "Model produceScore BoosterUpdateOneIter");
            if (subvalid != nullptr)
                LGBM_BoosterGetEval(booster, 1, &out_len, out_results);
            else
                LGBM_BoosterGetEval(booster, 0, &out_len, out_results);
            score = out_results[0];
            if (earlyStop(score, 200, earlystopcount, bestscore))break;
            if(abs(score - bestscore) < 1e-20)bestiter = iter + 1;
            if (is_finished) break;
        }
        LOG(INFO) << "The best score of the model: " << bestscore;
        m_init_loss = bestscore;
        delete [] out_results;
        int featurenums;
        int orifeaturenums = m_fcdataset->getFeatureSize(false);
        LGBM_BoosterGetNumFeature(booster, &featurenums);
        out_results = new double[featurenums];
        auto importance_types = atoi(Property::getProperty()->getLightGBMParams().find("important_type")->second.c_str());
        int importance_type = importance_types & 1;
        int isnotbestiter_type = importance_types & 2;
        if(isnotbestiter_type)bestiter = 0;
        LOG(DEBUG) << "the importance type: " << importance_type << " the best iter: " << bestiter;
        LGBM_BoosterFeatureImportance(booster, bestiter, importance_type, out_results);
        CCHECK_EQ(featurenums, orifeaturenums, "Model initdata featurenums != orifeaturenums");
        memcpy(scores.data(), out_results, featurenums * sizeof(double));
        delete[] out_results;
        LGBM_BoosterFree(booster);
        LGBM_DatasetFree(subtrain);
        LGBM_DatasetFree(subvalid);
    }

    DatasetHandle Model::getSubDataset(DatasetHandle dataset, const std::vector<int>& instanceindex)
    {
        if(dataset == nullptr)
            return nullptr;
        DatasetHandle subdataset;
        string parameters("");
        int flag = LGBM_DatasetGetSubset(dataset, instanceindex.data(), instanceindex.size(), parameters.c_str(), &subdataset);
        checkStatus(flag, "Model getSubDataset LGBM_DatasetGetSubset");
        return subdataset;
    }

    DatasetHandle Model::generateLightGBMDatasetFromMat(const vector<int>& featureindex)
    {
        auto numclass = Property::getProperty()->getTargetClasses();
        MyDataType *datavalues;
        auto tempdataset = m_fcdataset;
        auto numinstances= tempdataset->getInstancesOfFeature();
        
        auto orifeatureinfos = tempdataset->getFeatures();
        CCHECK_GE(orifeatureinfos.size(), 1, "Model generateLightGBMDatasetFromMat orifeatureinfos.size() < 1");
        CCHECK_GE(orifeatureinfos.size(), featureindex.size(), "Model generateLightGBMDatasetFromMat orifeatureinfos.size() < featureindex.size()");
        int featurenums = orifeatureinfos.size();
        // cout << "featurenums: " << featurenums << endl;
		vector<int> selectfeatureindex(0);
        selectfeatureindex.insert(selectfeatureindex.end(), featureindex.begin(), featureindex.end());
        if (featureindex.size() == 0)
        {
            selectfeatureindex.clear();
            for (int i = 0; i < featurenums; ++i) selectfeatureindex.push_back(i);
        }
        else featurenums = selectfeatureindex.size();
        datavalues = new MyDataType[featurenums*numinstances];
        string categorical_features = "categorical_feature=";
        
        for(int index = 0; index < selectfeatureindex.size(); ++index)
        {
            int i = selectfeatureindex[index];
            CCHECK_LT(i, orifeatureinfos.size(), "Model generateLightGBMDatasetFromMat selectfeatureindex[i] >= orifeatureinfos.size()");
            orifeatureinfos[i] = tempdataset->getFeatureFromOld(orifeatureinfos[i]);
            auto featurevalues = orifeatureinfos[i]->getFeature()->getValues();
            if(orifeatureinfos[i]->getType() == OutType::Discrete)
            {
                categorical_features += std::to_string(index) + ",";
                for(int j = 0;j < numinstances; ++j)
                    datavalues[index * numinstances + j] = static_cast<MyDataType>(reinterpret_cast<int*>(featurevalues)[j]);
            }
            else
                memcpy(datavalues + index * numinstances, reinterpret_cast<MyDataType*>(featurevalues), numinstances * sizeof(MyDataType));
            tempdataset->clearFeatureData(orifeatureinfos[i]);
        }
       
        if (categorical_features != "categorical_feature=")
            categorical_features.pop_back(), categorical_features += " ";
        else categorical_features = "";
        string parameters;
        // getParams(parameters);
        parameters += categorical_features;
        DatasetHandle dataset;
        auto datatype = sizeof(MyDataType) == 4 ? C_API_DTYPE_FLOAT32 : C_API_DTYPE_FLOAT64;
        int flag = LGBM_DatasetCreateFromMat(datavalues, datatype, numinstances, featurenums, 0, parameters.c_str(), nullptr, &dataset);
        int targetindex = tempdataset->getDatasetIndex()[tempdataset->getID()];
        
        auto featurevalues = tempdataset->getTargetFeature()->getFeature()->getValues();
        float *targetvalues = new float[numinstances];
        if(tempdataset->getTargetFeature()->getType() == OutType::Discrete)
        {
            for (int i = 0; i < numinstances; ++i)
                targetvalues[i] = static_cast<float>(reinterpret_cast<int*>(featurevalues)[i+targetindex]);
        }
        else
            for (int i = 0; i < numinstances; ++i)
                targetvalues[i] = static_cast<float>(reinterpret_cast<MyDataType*>(featurevalues)[i+targetindex]);
        int hasinit = 1;
        double* init_scores = m_init_scores.size() == 0 ? nullptr : m_init_scores.data();
        if (init_scores == nullptr) hasinit = 0;
        numclass = numclass <= 2 ? 1 : numclass;
        flag = LGBM_DatasetInitStreaming(dataset, 0, hasinit, 0, numclass, 60, -1);
        flag = LGBM_DatasetPushRowsWithMetadata(dataset, datavalues, datatype, numinstances, featurenums, 0, targetvalues, nullptr, init_scores, nullptr, 0);
        flag = LGBM_DatasetMarkFinished(dataset);
        checkStatus(flag, "Model generateLightGBMDatasetFromMat LGBM_DatasetCreateFromSampledColumn");
       
        try{
            delete[] datavalues;
            delete[] targetvalues;
            targetvalues = nullptr;
            datavalues = nullptr;
        }catch(std::exception & e)
        {
            LOG(WARNING) << e.what();
        }
        return dataset;
    }

    void Model::getParams(std::string &parameters)
    {
        auto numclass = Property::getProperty()->getTargetClasses();
        auto &lightgbmparams = Property::getProperty()->getLightGBMParams();
        if (numclass > 2)
            parameters += "objective=multiclass metric=multi_error num_class=" + std::to_string(numclass);
        else if(numclass == 2)
            parameters += "objective=binary metric=binary_error";
        else
            parameters += "objective=regression metric=rmse";
        for (auto& param : lightgbmparams)
        {
            if (param.first != "important_type")parameters += " " + param.first + "=" + param.second;
            if(m_featureindex.size() == 0 && param.first == "num_threads")
                LGBM_SetMaxThreads(std::stoi(param.second));
        }
    }

    BoosterHandle Model::InitBooster(DatasetHandle train_data, DatasetHandle valid_data)
    {
        BoosterHandle booster = nullptr;
        string parameters;
        getParams(parameters);
        int flag = LGBM_BoosterCreate(train_data, parameters.c_str(), &booster);    
        checkStatus(flag, "Model InitBooster BoosterCreate");
        if(valid_data != nullptr)
        {
            flag = LGBM_BoosterAddValidData(booster, valid_data);
            checkStatus(flag, "Model produceScore BoosterAddValidData");
        }
        return booster;
    }

    void Model::checkStatus(int flag, const std::string & message) {
        if (flag != 0) {
            LOG(ERROR) << message << " error : " << flag;
            exit(flag);
        }
    }

    void Model::clear()
    {
        // if(m_traindata != nullptr)
        // {
        //     LGBM_DatasetFree(m_traindata);
        //     m_traindata = nullptr;
        // }
        // if(m_validdata != nullptr)
        // {
        //     LGBM_DatasetFree(m_validdata);
        //     m_validdata = nullptr;
        // }
        m_init_scores.clear();
    }

    const std::vector<double>& Model::getInitScores()
    {
        return m_init_scores;
    }

    const std::vector<int>& Model::getFeatureIndex()
    {
        return m_featureindex;
    }

    bool Model::earlyStop(double score, int earlystopround, int& earlystopcount, double& bestscore)
    {
        if (earlystopround <= 0)
            return false;
        if (bestscore - score > 1e-5)
        {
            earlystopcount = 0;
            bestscore = std::min(score, bestscore);
            return false;
        }
        else
        {
            earlystopcount++;
            if (earlystopcount >= earlystopround)
            {
                LOG(DEBUG) << "The early stop of the model, and the early stop round is: " << earlystopcount;
                return true;
            }
            return false;
        }
    }

    string Model::stateToString(const std::vector<unsigned int>& state)
    {
        string result = "";
        for (unsigned int i = 0; i < state.size(); i++)
        {
            result += std::to_string(state[i]) + '_';
        }
        return result;
    }

    const std::vector<unsigned int> Model::stringToState(const string& state)
    {
        std::vector<unsigned int> result;
        std::stringstream ss(state);
        string item;
        while (std::getline(ss, item, '_'))
        {
            result.push_back(std::stoi(item));
        }
        return result;
    }
}//RL4FS