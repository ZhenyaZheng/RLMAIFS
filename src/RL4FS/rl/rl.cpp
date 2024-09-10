#include "rl/rl.h"
#include <cmath>

namespace RL4FS{
    ReinforcementLearning::ReinforcementLearning(DataSet *dataset, Params* param):m_qlearning (QLearning(param->gamma, param->alpha, param->epsilon))
    {
        m_gamma = param->gamma;
        m_alpha = param->alpha;
        m_epsilon = param->epsilon;
        m_maxIter = param->n_search_iterations;
        m_maxIterParticle = param->maxIterParticle;
        m_numofParticles = param->n_individuals;
        m_dataset = dataset;
        int numid = 0;
        auto &state = param->state;
        m_model = std::make_shared<Model>(dataset);
        param->feature_size = dataset->getFeatureSize(false) - m_model->getFeatureIndex().size();
        m_bestresult = new Result(std::vector<unsigned int>(param->feature_size, 0), m_model, param);
        m_earlystop = param->earlystopiterations;
        for(int i = 0; i < m_numofParticles; i++)
        {
            if (state.size() > 0)
            {
                LOG(INFO) << "Using the state from the input: " << state;
                m_particles.push_back(new Particle(m_model, state, numid++, m_maxIterParticle, &m_qlearning, param));
            }
            else
            {
                // random select a state
                std::vector<unsigned int> state(param->feature_size, 0);
                for (int j = 0; j < param->feature_size; j++)
                {
					std::random_device rd;
					std::mt19937 gen(rd());
					auto dist = std::uniform_int_distribution<unsigned int>(0, 1);
					state[j] = dist(gen);
				}
                m_particles.push_back(new Particle(m_model, state, numid++, m_maxIterParticle, &m_qlearning, param));
            }
        }
    }

    ReinforcementLearning::~ReinforcementLearning()
    {
        if (m_bestresult != nullptr)
            delete m_bestresult;
        m_bestresult = nullptr;
        for(auto &particle : m_particles)
        {
            if (particle != nullptr)
                delete particle;
            particle = nullptr;
        }
    }

    vector<int> ReinforcementLearning::optimize()
    {
        const int threadnums = Property::getProperty()->getThreadNum();
        int earlystop = 0;
        for(int i = 0; i < m_maxIter; i++)
        {
            LOG(INFO) << "Running iteration " << i << " of " << m_maxIter;
            // omp_set_nested(1);
            // #pragma omp parallel for num_threads(threadnums)
            std::vector<std::thread> threadspool;
            for(int j = 0; j < m_particles.size(); j++)
            {
                // m_particles[j]->run(m_bestresult);
                threadspool.push_back(std::thread(&Particle::run, m_particles[j], m_bestresult));
            }
            for(auto &thread : threadspool)
            {
                thread.join();
            }
            bool hasbest = false;
            for(int j = 0; j < m_particles.size(); j++)
            {   auto bestloss = m_particles[j]->getLoss();
                LOG(INFO) << "Particle " << j << " loss: " << bestloss;
                if (bestloss < m_bestloss)
                {
                    hasbest = true;
                    m_bestloss = bestloss;
                    if(m_bestresult != nullptr) delete m_bestresult;
                    m_bestresult = m_particles[j]->getResult();
                    LOG(INFO) << "Found new best loss: " << bestloss << " and the state is " << m_bestresult->m_beststate;
                }
            }
            if (!hasbest)
            {
                earlystop++;
                if (m_earlystop > 0 && earlystop >= m_earlystop)
                {
                    LOG(INFO) << "Early stop at iteration " << i;
                    break;
                }
            }
            else
                earlystop = 0;
        }
        if(m_bestresult == nullptr)
        {
            LOG(INFO) << "The best result is null";
            return m_model->getFeatureIndex();
        }
        if (m_bestloss > m_model->getScore())
        {
            LOG(INFO) << "The best result is not the best";
            vector<int> result;
            if(m_model->getInitLoss() < m_model->getScore())
            {
                LOG(INFO) << "The init loss is better than the current loss";
                m_bestloss = m_model->getInitLoss();
                for (int i = 0; i < m_dataset->getFeatureSize(false); i++)
                    result.push_back(i);
                return result;
            }
            m_bestloss = m_model->getScore();
            result = m_model->getFeatureIndex();
            sort(result.begin(), result.end());
            return result;
        }
        auto &beststate = m_bestresult->m_beststate;
        std::vector<int> result(m_model->getFeatureIndex());
        for (int i = 0; i < beststate.size(); i++)
        {
            if (beststate[i]) result.push_back(m_model->m_featureindexmap[i]);
		}
        std::sort(result.begin(), result.end());
        return result;
    }

    double ReinforcementLearning::getBestLoss()
    {
		return m_bestloss;
	}
} // namespace RL4FS
