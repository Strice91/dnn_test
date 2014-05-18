#include "Neuron.h"

Neuron::Neuron(int const index_,
	std::function<double(double, double)> const input_function_,
	std::function<double(double)> const output_function_) :
index(index_), input(false), output(false),
current_input_function(input_function_),
current_output_function(output_function_)
{

}

Neuron::~Neuron()
{

}

size_t Neuron::get_index() const
{
	return index;
}


double Neuron::input_function(double prev_, double curr_) const
{
	return current_input_function(prev_, curr_);
}

double Neuron::output_function(double input_) const
{
	return current_output_function(input_);
}

double Neuron::default_input_function(double prev_, double curr_)
{
	return prev_ + curr_;
}

double Neuron::default_output_function(double value_)
{
	return tanh(value_);
}

bool Neuron::is_input() const
{
	return input;
}

bool Neuron::is_output() const
{
	return output;
}

void Neuron::set_as_input()
{
	input = true;
}

void Neuron::set_as_output()
{
	output = true;
}

void Neuron::set_input_function(std::function<double(double, double)> input_function_)
{
	current_input_function = input_function_;
}

void Neuron::set_output_function(std::function<double(double)> output_function_)
{
	current_output_function = output_function_;
}
