#ifndef NEURON_H
#define NEURON_H

#include <functional>
#include <stdlib.h>
#include <math.h>

class Neuron
{
	public:
		explicit Neuron(int const index_,
			std::function<double(double, double)> const current_input_function = default_input_function,
			std::function<double(double)> const current_output_function = default_output_function);
		virtual ~Neuron();

		bool is_input() const;
		bool is_output() const;

		void set_as_input();
		void set_as_output();
		void set_input_function(std::function<double(double, double)> input_function_);
		void set_output_function(std::function<double(double)> output_function_);

		double input_function(double prev_, double curr_ = 0) const;
		double output_function(double value_) const;

		size_t get_index() const;

	private:
		int index;
		bool output;
		bool input;
		std::function<double(double, double)> current_input_function;
		std::function<double(double)> current_output_function;
		static double default_input_function(double prev_, double curr_);
		static double default_output_function(double value_);
};

#endif
