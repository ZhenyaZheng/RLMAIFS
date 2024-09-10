#pragma once
#include "RL4FS/util/log.h"
#include "qlearning.h"
#include "particle.h"
#include "RL4FS/DataSet.h"
#include "model.h"
#include <limits>
namespace RL4FS{
    class ReinforcementLearning
    {
        QLearning m_qlearning;
        Result *m_bestresult = nullptr;
        float_type m_gamma;
        float_type m_alpha;
        float_type m_epsilon;
        int m_maxIter;
        int m_maxIterParticle;
        int m_numofParticles;
        DataSet *m_dataset = nullptr;
        std::shared_ptr<Model> m_model = nullptr;
        std::vector<Particle*> m_particles;
        float_type m_bestloss = (std::numeric_limits<float_type>::max)();
        int m_earlystop = 0;
    public:
        
        ReinforcementLearning(DataSet *dataset, Params* param);
        ~ReinforcementLearning();
        double getBestLoss();
        vector<int> optimize();
    };
}