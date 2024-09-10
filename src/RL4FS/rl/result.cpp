#include "rl/result.h"

namespace RL4FS{
    Result::~Result()
    {
        m_results.clear();
    }

    Result::Result(const std::vector<unsigned int>& state,std::shared_ptr<Model>  model, Params* param)
    {
        m_state = state;
        m_model = model;
        m_param = param;
        m_results.clear();
        m_beststate = state;
        m_bestloss = getBestLoss();
    }

    Result::Result(const Result& result)
    {
        m_results = result.m_results;
        m_beststate = result.m_beststate;
        m_state = result.m_state;
        m_model = result.m_model;
        m_param = result.m_param;
        m_bestloss = result.m_bestloss;
    }

    Result& Result::operator=(Result* && presult)
    {
        if (presult == nullptr || this == presult)
			return *this;
        m_state = presult->m_state;
        m_model = presult->m_model;
        m_param = presult->m_param;
        m_bestloss = presult->m_bestloss;
        m_results = presult->m_results;
        m_beststate = presult->m_beststate;
        return *this;
    }

    void Result::setLoss(float_type value)
    {
        string state = Model::stateToString(m_state);
        m_results[state] = value;
    }

    float_type Result::getLoss()
    {
        string state = Model::stateToString(m_state);
        if (m_results.find(state) == m_results.end())
        {
            auto loss = m_model->getLoss(m_state);
            if (std::isnan(loss) || std::isinf(loss))
                m_results[state] = std::numeric_limits<float_type>::max();
            else m_results[state] = loss;
        }
        return m_results[state];
    }

    float_type Result::getLoss(const std::vector<unsigned int>& state)
    {
        string stateStr = Model::stateToString(state);
        if (m_results.find(stateStr) == m_results.end())
        {
            auto loss = m_model->getLoss(state);
            if (std::isnan(loss) || std::isinf(loss))
                m_results[stateStr] = std::numeric_limits<float_type>::max();
            else m_results[stateStr] = loss;
        }
        return m_results[stateStr];
    }

    float_type Result::getBestLoss()
    {
        return getLoss(m_beststate);
    }

    const std::vector<unsigned int>& Result::getBestState()
    {
        return m_beststate;
    }

    void Result::setBestState(const std::vector<unsigned int>& state)
    {
        m_beststate = state;
    }

} // namespace RL4FS

