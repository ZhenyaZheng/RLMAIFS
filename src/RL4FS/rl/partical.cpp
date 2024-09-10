#include "rl/particle.h"

namespace RL4FS{
    Particle::~Particle()
    {
        if (m_presult != nullptr)
            delete m_presult;
        m_presult = nullptr;
    }

    Particle::Particle(std::shared_ptr<Model> model, const std::vector<unsigned int> &state, int id, int maxiter, QLearning *qlearning, Params *param):m_param(param)
    {
        m_id = id;
        m_maxiter = maxiter;
        m_qlearning = qlearning;
        m_presult = new Result(state, model, param);
    }

    void Particle::run(Result *gbest_result)
    {
        LOG(INFO) << "Particle " << m_id << " begin running!";
        int earlystopround = m_param->earlystopround;
        int earlystop = 0;
        for (int i = 0; i < m_maxiter; i++)
        {
            LOG(DEBUG) << "Particle " << m_id << " running iteration " << i << " of " << m_maxiter;
            auto & state = m_presult->m_state;
            auto beststate = m_presult->m_beststate;
            std::vector<Action> actions;
            for(unsigned int j = 0;j < state.size(); j++)
            {
                bool flag = false;
                auto epsilon = rand() / (1.0 + RAND_MAX);
                if(epsilon < 0.5)
                {
                    if (gbest_result != nullptr)
                    {
                        auto &gbeststate = gbest_result->m_beststate;
                        flag = true;
                        if (state[j] != gbeststate[j])
                            actions.push_back({j, gbeststate[j]});
                    }                
                }
                else if (epsilon < 0.9)
                {
                    if(beststate.size() > 0)
                    {
                        CHECK_EQ(beststate.size(), state.size());
                        flag = true;
                        if (state[j] != beststate[j])
                            actions.push_back({j, beststate[j]});
                    }
                }
                if(!flag || epsilon >= 0.9)
                {
                    auto value = (state[j] + 1) % 2;
                    actions.push_back({j, value});
                }
            }
            std::pair<std::vector<unsigned int>, Action> state_action;
            state_action.first = state;
            m_qlearning->getNext(m_presult, actions, state_action);
            auto &newstate = state_action.first;
            auto &action = state_action.second;
            Result *newpresult = new Result(newstate, m_presult->m_model, m_param);
            LOG(DEBUG) << "Particle " << m_id << " running iteration " << i << " of " << m_maxiter << " getloss";
            auto oldloss = m_presult->getLoss();
            auto newloss = newpresult->getLoss();
            LOG(DEBUG) << "Particle " << m_id << " running iteration " << i << " of " << m_maxiter << " getloss done";
            float_type reward = 0;
            auto NumZero = [&](const std::vector<unsigned int> &state)
            {
                auto num = 0;
                for (auto &s : state)
                {
                    if (s == 0)
                        num++;
                }
                return num;
            };
            if (newloss > oldloss)
                reward = oldloss - newloss;
            else if (newloss <= oldloss)
            {
                if(newloss < oldloss)
                    reward = oldloss + newloss;
                else if (NumZero(newstate) > NumZero(state))
                    reward = newloss;
                else reward = newloss / 2;
            }
            m_qlearning->updateQValue(m_presult, newpresult, actions, action, reward);
            m_presult->m_state = newpresult->m_state;
            m_presult->setLoss(newloss);
            delete newpresult;
            if (newloss < m_presult->getBestLoss())
            {
                m_presult->m_beststate = m_presult->m_state;
                m_presult->m_bestloss = newloss;
                earlystop = 0;
            }
            else
            {
                earlystop++;
                if (earlystop >= earlystopround)
                {
                    LOG(INFO) << "Particle " << m_id << " early stop, iteration " << i;
                    break;
                }
            }
        }
        LOG(INFO) << "Particle " << m_id << " finished!";
    }

    float_type Particle::getLoss()
    {
        return m_presult->getBestLoss();
    }

    Result* Particle::getResult()
    {
        return  new Result(*m_presult);
    }
}// namespace RL4FS

