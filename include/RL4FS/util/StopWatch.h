#ifndef __STOPWATCH_H__
#define __STOPWATCH_H__
#include <chrono>

namespace RL4FS {
	class StopWatch
	{
	public:

		StopWatch();
		~StopWatch();

		void Start();

		void Stop();

		void ReStart();

		double Elapsed();

		double ElapsedMS();

		double ElapsedSecond();

	private:
		long long elapsed_;
		typedef std::chrono::steady_clock Clock;
		typedef std::chrono::microseconds MicroSeconds;
		std::chrono::steady_clock::time_point start_;
		std::chrono::steady_clock::time_point stop_;

	};
}
#endif // __STOPWATCH_H__




