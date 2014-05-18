#ifndef RANDOM_H
#define RANDOM_H

#include <chrono>
#include <random>

namespace random 
{
	template<typename T> T int_in_range(T const &lower, T const &upper)
	{
		std::uniform_int_distribution<T> random_int(lower, upper);
		static std::default_random_engine random_engine(std::random_device{}());
		return random_int(random_engine);
	}

	template<typename T> T float_in_range(T const &lower, T const &upper)
	{
		std::uniform_real_distribution<T> random_double(lower, upper);
		static std::default_random_engine random_engine(std::random_device{}());
		return random_double(random_engine);
	}
	
}

#endif