#include "RL4FS/util/StopWatch.h"
namespace RL4FS {

	StopWatch::StopWatch() :elapsed_(0), start_(MicroSeconds::zero()), stop_(MicroSeconds::zero())
	{
		Start();
	}
	StopWatch::~StopWatch()
	{
	}
	
	void StopWatch::Start()
	{
		start_ = Clock::now();
	}

	void StopWatch::Stop()
	{
		stop_ = Clock::now();
		elapsed_ += std::chrono::duration_cast<MicroSeconds>(stop_ - start_).count();
	}

	void StopWatch::ReStart()
	{
		elapsed_ = 0;
		Start();
	}

	double StopWatch::Elapsed()
	{
		return static_cast<double>(elapsed_);
	}

	double StopWatch::ElapsedMS()
	{
		return elapsed_ / 1000.0;
	}

	double StopWatch::ElapsedSecond()
	{
		return elapsed_ / 1000000.0;
	}
}
