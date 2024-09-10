#pragma once
#include "rl/result.h"
#include <random>
#include <mutex>
#include <shared_mutex>
namespace RL4FS{
    struct Action
    {
        unsigned int m_index;
        unsigned int m_value;
        Action() : m_index(0), m_value(0) {};
        Action(unsigned int index, unsigned int value) : m_index(index), m_value(value) {};
        string toString() const
        {
            return std::to_string(m_index) + " " + std::to_string(m_value);
        }
    };

    class QLearning
    {
        
        std::unordered_map<string, std::unordered_map<string, float_type>> m_qValues;
        float_type m_gamma;
        float_type m_alpha;
        float_type m_epsilon;
        std::shared_timed_mutex* m_rwmutex = nullptr;
    public:
            
            ~QLearning();
            QLearning(float_type gamma, float_type alpha, float_type epsilon);
            
            void updateQValue(Result* presult, Result* pnewresult, std::vector<Action>& actions, Action& action, float_type reward);
            
            float_type getQValue(string state, Action action);

            float_type getBestQValue(Result* presult, std::vector<Action>& actions, Action &maxaction);

            void setQValue(string state, Action action, float_type value);

            void getNext(Result* presult, std::vector<Action>& actions, std::pair<std::vector<unsigned int>, Action>&);

            void getNewState(const std::vector<unsigned int> &state, std::vector<unsigned int> &newstate, Action action);
    };
}