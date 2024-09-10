#pragma once
#include "model.h"
#include <limits>
#include <cmath>
namespace RL4FS{

    struct Result
    {
        std::unordered_map<string, float_type> m_results;
        std::vector<unsigned int> m_beststate;
        std::vector<unsigned int> m_state;
        Params* m_param = nullptr;
        float_type m_bestloss = (std::numeric_limits<float_type>::max)();
        std::shared_ptr<Model> m_model = nullptr;

        ~Result();
        Result(const std::vector<unsigned int>& state, std::shared_ptr<Model> model, Params* param);
        Result(const Result& result);
        Result& operator=(Result* &&);
        float_type getLoss();
        float_type getLoss(const std::vector<unsigned int>& state);
        float_type getBestLoss();
        void setLoss(float_type value);
        const std::vector<unsigned int>& getBestState();
        void setBestState(const std::vector<unsigned int>& state);
    };
}
