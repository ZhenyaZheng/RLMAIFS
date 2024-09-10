#include "rl/result.h"
#include "rl/qlearning.h"
namespace RL4FS{
    class Particle
    {
        int m_id;
        Result* m_presult = nullptr;
        int m_maxiter;
        QLearning *m_qlearning = nullptr;
        Params *m_param = nullptr;
    public:
        Particle() = delete;
        ~Particle();
        Particle(std::shared_ptr<Model> model, const std::vector<unsigned int> &state, int id, int maxiter, QLearning *qlearning, Params *param);
        void run(Result *bestresult);
        float_type getLoss();
        Result* getResult();
    };
}
    
