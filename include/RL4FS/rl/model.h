#pragma once
#include "RL4FS/DataSet.h"
#include "LightGBM/c_api.h"
#include <unordered_set>
namespace RL4FS{
#ifdef USE_DOUBLE
    using float_type = double;
#else
    using float_type = float;
#endif
    struct Params
    {
        float_type gamma = 0.9;
        float_type alpha = 0.1;
        float_type epsilon = 0.9;
        vector<unsigned int> state;
        int n_search_iterations = 10;
        int maxIterParticle = 10;
        int n_individuals = 10;
        int feature_size = 0;
        int earlystopround = 30;
        int earlystopiterations = 5;
        Params() {};
        Params(const std::unordered_map<string, string>& params) {
			if (params.find("gamma") != params.end())
				gamma = static_cast<float_type>(std::stof(params.at("gamma")));
		    if (params.find("alpha") != params.end())
                alpha = static_cast<float_type>(std::stof(params.at("alpha")));
            if (params.find("epsilon") != params.end())
                epsilon = static_cast<float_type>(std::stof(params.at("epsilon")));
            if (params.find("n_search_iterations") != params.end())
                n_search_iterations = std::stoi(params.at("n_search_iterations"));
            if (params.find("maxIterParticle") != params.end())
                maxIterParticle = std::stoi(params.at("maxIterParticle"));
            if (params.find("n_individuals") != params.end())
                n_individuals = std::stoi(params.at("n_individuals"));
            if (params.find("state") != params.end())
                state = stringToState(params.at("state"));
            if (params.find("feature_size") != params.end())
                feature_size = std::stoi(params.at("feature_size"));
            if (params.find("earlystopround") != params.end())
                earlystopround = std::stoi(params.at("earlystopround"));
            if (params.find("earlystopiterations") != params.end())
                earlystopiterations = std::stoi(params.at("earlystopiterations"));
		}
        Params(int n_search_iterations, int n_individuals = 10, int maxIterParticle = 10, float_type gamma = 0.9, float_type alpha = 0.1, float_type epsilon = 0.9, const vector<unsigned int>& state= vector<unsigned int>(0), int earlystop = 30, int earlystopiterations=5) : gamma(gamma), alpha(alpha), epsilon(epsilon), state(state), n_search_iterations(n_search_iterations), maxIterParticle(maxIterParticle), n_individuals(n_individuals), earlystopround(earlystop), earlystopiterations(earlystopiterations) {}
        std::vector<unsigned int> stringToState(const string& state)
        {
			std::vector<unsigned int> result;
			std::string temp;
            for (auto c : state)
            {
                if (c == '[' || c == ']' || c == ',' || c == ' ')
                {
                    if (temp.size() > 0)
                    {
						result.push_back(std::stoi(temp));
						temp.clear();
					}
				}
				else
					temp.push_back(c);
			}
			return result;
		}
    };


    class Model
    {
        public:
            ~Model();
            Model(DataSet* dataset);
            double getLoss(const vector<unsigned int>& state);
            void clear();
            const std::vector<double>& getInitScores();
            const std::vector<int>& getFeatureIndex();
            std::unordered_map<int, int> m_featureindexmap;
            static string stateToString(const std::vector<unsigned int>& state);
            static const std::vector<unsigned int> stringToState(const string& state);
            double getInitLoss();
            double getScore();
    private:
            void initdata();
            DatasetHandle generateLightGBMDatasetFromMat(const vector<int>& featureindex=vector<int>(0));
            DatasetHandle getSubDataset(DatasetHandle dataset, const std::vector<int>& instanceindex);
            BoosterHandle InitBooster(DatasetHandle train, DatasetHandle valid);
            void getParams(std::string &parameters);
            bool earlyStop(double score, int earlystopround, int& earlystopcount, double& bestscore);
            void checkStatus(int flag, const std::string & message);
            void selectImportantFeature(vector<double> &scores);

            // DatasetHandle m_traindata;
            // DatasetHandle m_validdata;
        protected:
            DataSet* m_fcdataset = nullptr;
            vector<int> m_featureindex;
            vector<double> m_init_scores;
            double m_score = (std::numeric_limits<double>::max)();
            double m_init_loss = (std::numeric_limits<double>::max)();
            std::unordered_map<string, double> m_statescores;
            // lock for m_statescores
            std::mutex m_statescores_lock;
    };
}