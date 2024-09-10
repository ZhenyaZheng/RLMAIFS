#include "rl/qlearning.h"

namespace RL4FS{
    QLearning::~QLearning()
    {
        if (m_rwmutex != nullptr)
            delete m_rwmutex;
        m_rwmutex = nullptr;
    }

    QLearning::QLearning(float_type gamma, float_type alpha, float_type epsilon)
    {
        m_gamma = gamma;
        m_alpha = alpha;
        m_epsilon = epsilon;
        if (m_rwmutex == nullptr)
            m_rwmutex = new std::shared_timed_mutex();
    }

    void QLearning::updateQValue(Result* presult, Result* pnewresult, std::vector<Action>& actions, Action& action, float_type reward)
    {
        CHECK_NE(presult, nullptr);
        CHECK_NE(pnewresult, nullptr);
        Action maxaction;
        auto qvalue = getBestQValue(pnewresult, actions, maxaction);
        qvalue = reward + m_gamma * qvalue;

        auto state = Model::stateToString(presult->m_state);
        
        auto actionStr = action.toString();
        auto newqvalue = (1 - m_alpha) * getQValue(state, action) + m_alpha * qvalue;
        m_rwmutex->lock();
        if(m_qValues.find(state) == m_qValues.end())
            m_qValues[state] = std::unordered_map<string, float_type>();
        m_qValues[state][actionStr] = newqvalue;
        m_rwmutex->unlock();
    }

    float_type QLearning::getQValue(string state, Action action)
    {
        auto actionStr = action.toString();
        float_type qvalue = 0;
        m_rwmutex->lock_shared();
        if (m_qValues.find(state) != m_qValues.end() && m_qValues[state].find(actionStr) != m_qValues[state].end())
            qvalue = m_qValues[state][actionStr];
        m_rwmutex->unlock_shared();
        return qvalue;
    }

    float_type QLearning::getBestQValue(Result* presult, std::vector<Action>& actions, Action &maxaction)
    {
        auto maxQValue = 0;
        std::vector<size_t> indexes;
        /*if (actions.size() > 20)
        {
            std::shuffle(actions.begin(), actions.end(), std::mt19937(std::random_device()()));
            actions.resize(20);
        }*/
        for (size_t i = 0; i < actions.size();++ i)
        {
            auto nextstate(presult->m_state);
            getNewState(presult->m_state, nextstate, actions[i]);
            auto qvalue = getQValue(Model::stateToString(presult->m_state), actions[i]);
            
            if (qvalue > maxQValue)
            {
                maxQValue = qvalue;
                indexes.clear();
                indexes.push_back(i);
            }
            else if (qvalue == maxQValue)
            {
                indexes.push_back(i);
            }
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned> uniform_dist(0, indexes.size() - 1);
        auto index = uniform_dist(gen);
        maxaction = actions[indexes[index]];
        return maxQValue;
    }

    void QLearning::setQValue(string state, Action action, float_type value)
    {
        auto actionStr = action.toString();
        if(m_qValues.find(state) == m_qValues.end())
            m_qValues[state] = std::unordered_map<string, float_type>();
        m_qValues[state][actionStr] = value;
    }

    void QLearning::getNext(Result* presult, std::vector<Action>& actions, std::pair<std::vector<unsigned int>, Action>& stateaction)
    {
        CHECK_NE(presult, nullptr);
        auto& newstate = stateaction.first;
        auto& newaction = stateaction.second;
        if (this->m_epsilon > rand() / (1.0 + RAND_MAX))
        {
            // random select a action
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<unsigned> uniform_dist(0, actions.size() - 1);
            auto index = uniform_dist(gen);
            newaction = actions[index];
            getNewState(presult->m_state, newstate, newaction);
        }
        else 
        {
            auto bestloss = getBestQValue(presult, actions, newaction);
            getNewState(presult->m_state, newstate, newaction);
        }
    }

    void QLearning::getNewState(const std::vector<unsigned int> &state, std::vector<unsigned int> &newstate, Action action)
    {
        auto index = action.m_index;
        auto newval = action.m_value;
        CHECK_LE(newval, 1);
        CHECK_LE(index, state.size() - 1);
        CHECK_EQ(state.size(), newstate.size());
        newstate[index] = newval;
    }
}// namespace RL4FS
