#include <iostream>
#include <algorithm>
#include "ANN.h"
#include "random.h"

ANN::ANN(size_t const &neuron_count_) :
neuron_count(neuron_count_), sort_required(false), input_count(0), output_count(0), connection_count(0)
{
	for (size_t i = 0; i < neuron_count; i++) {
		neurons.push_back(Neuron(i));
	}
	biases.resize(neuron_count);
	weights.resize(neuron_count);
	connections.resize(neuron_count);
	for (auto &i : weights) {
		i.resize(neuron_count);
	}
	for (auto &i : connections) {
		i.resize(neuron_count);
	}
	sorted_indices.reserve(neuron_count);
	for (size_t i = 0; i < neuron_count; i++) {
		sorted_indices.push_back(i);
	}
}

ANN::~ANN()
{

}


double ANN::TransferFunctions::identity(double d)
{
	return d;
}

double ANN::TransferFunctions::heaviside(double d)
{
	if (d < 0)
		return 0;
	return 1;
}

double ANN::TransferFunctions::sigmoid(double d)
{
	return 1/(1 + exp(-d));
}

double ANN::TransferFunctions::ReLU(double d)
{
	if (d < 0)
        return 0;
    else
        return d;
}

double ANN::TransferFunctions::tangent_hyperbolic(double d)
{
	return tanh(d);
}

void ANN::display_connections() const
{
	for (size_t i = 0; i < neuron_count; i++) {
		for (size_t j = 0; j < neuron_count; j++) {
			std::cout << connections[i][j] << ' ';
		}
		std::cout << '\n';
	}
}

void ANN::display_weights() const
{
	for (size_t i = 0; i < neuron_count; i++) {
		for (size_t j = 0; j < neuron_count; j++) {
			std::cout << weights[i][j] << ' ';
		}
		std::cout << '\n';
	}
}

void ANN::display_biases() const
{
	for (size_t i = 0; i < neuron_count; i++) {
		std::cout << biases[i] << '\n';
	}
}

bool ANN::compare_topology(size_t const &first, size_t const &second) const
{
	int value1 = 0, value2 = 0, currentValue = 1;

	for (size_t i = 0; i < neuron_count; i++) {
		if (connections[first][i])
			value1 = currentValue;
		if (connections[second][i])
			value2 = currentValue;
		currentValue++;
	}
	return value1 < value2;
}

void ANN::topological_sort() const
{
	std::sort(sorted_indices.begin(), sorted_indices.end(), [&](size_t const &first_, size_t const &second_) { return compare_topology(first_, second_); });
}

double ANN::calculate_single_output(double const &input_) const
{
	return calculate_output(std::vector<double>{input_}).front();
}

double ANN::calculate_single_output(std::vector<double> const &input_) const
{
	return calculate_output(input_).front();
}

std::vector<double> ANN::calculate_output(double const &input_) const
{
	return calculate_output(std::vector<double>{input_});
}


bool contains_element(std::vector<size_t> const &vec_, size_t const value_)
{
	return std::find(vec_.begin(), vec_.end(), value_) != vec_.end();
}


std::vector<double> ANN::calculate_output(std::vector<double> const &input_) const
{

	/*std::vector<size_t> indices;
	size_t current_index = 0, i;

	while (indices.size() != neuron_count) {
		for (i = 0; i < neuron_count; i++) {
			if (connections[current_index][i] && i > current_index) {
				if (!contains_element(indices, i)) {
					current_index = i;
					break;
				}
			}
		}
		if (i == neuron_count) {
			indices.push_back(current_index);
			current_index = 0;
			while (contains_element(indices, current_index)) {
				current_index++;
			}
		}
	}

	int h = 0;*/


	std::vector<double> output;
	std::vector<double> final_output;;
	output.resize(neuron_count);
	final_output.reserve(output_count);
	size_t counter = 0;

	if (sort_required) {
		sort_required = false;
		topological_sort();
	}

	for (size_t k = 0; k < neuron_count; k++) {
		size_t i = sorted_indices[k];
		bool input_detected = false;
		for (size_t j = 0; j < neuron_count; j++) {
			if (neurons[i].is_input() && !input_detected) {
				output[i] = input_[counter];
				counter++;
				input_detected = true;
			}
			if (connections[i][j]) {
				//output[i] += weights[i][j]*output[j];
				output[i] = neurons[i].input_function(output[i], weights[i][j]*output[j]);
			}
		}
		output[i] = neurons[i].output_function(output[i] + biases[i]);
	}

	for (size_t i = 0; i < neuron_count; i++) {
		if (neurons[i].is_output())
			final_output.push_back(output[i]);
	}

	return final_output;

	return std::vector<double>();
}

size_t ANN::get_neuron_count() const
{
	return neuron_count;
}

void ANN::reset_internal_memory()
{
	prev_output.resize(neuron_count);
}

void ANN::connect_neurons(size_t const first_, size_t const second_, double const weight_)
{
	if (!connections[second_][first_]) {
		sort_required = true;
		connection_count++;
	}
	connections[second_][first_] = true;
	weights[second_][first_] = weight_;
}

void ANN::disconnect_neurons(size_t const first_, size_t const second_)
{
	if (connections[second_][first_]) {
		sort_required = true;
		connection_count--;
	}
	connections[second_][first_] = false;
	weights[second_][first_] = 0.0;
}

void ANN::set_weight_by_index(size_t const index_, double const &value_)
{
	size_t counter = 0;
	for (size_t i = 0; i < neuron_count; i++) {
		for (size_t j = 0; j < neuron_count; j++) {
			if (connections[i][j]) {
				if (index_ == counter) {
					weights[i][j] = value_;
					return;
				}
				counter++;
			}
		}
	}

	for (size_t i = 0; i < neuron_count; i++) {
		if (index_ == counter) {
			biases[i] = value_;
		}
		counter++;
	}
}

void ANN::set_weights(std::vector<std::vector<double>> const &weights_)
{
	for (size_t i = 0; i < neuron_count; i++) {
		for (size_t j = 0; j < neuron_count; j++) {
			weights[i][j] = weights_[i][j];
		}
	}
}

void ANN::set_biases(std::vector<double> const &biases_)
{
	for (size_t i = 0; i < neuron_count; i++) {
		biases[i] = biases_[i];
	}
}

void ANN::init_weights_random()
{
	init_weights_random(-1, 1);
}

void ANN::init_weights_random(double const lower_bound_, double const upper_bound_)
{
	for (size_t i = 0; i < neuron_count; i++) {
		biases[i] = random::float_in_range<double>(lower_bound_, upper_bound_);
		for (size_t j = 0; j < neuron_count; j++) {
			if (connections[i][j])
				weights[i][j] = random::float_in_range<double>(lower_bound_, upper_bound_);
			else
				weights[i][j] = 0.0;
		}
	}
}

size_t ANN::get_input_count() const
{
	return input_count;
}

size_t ANN::get_output_count() const
{
	return output_count;
}

size_t ANN::get_connection_count() const
{
	return connection_count;
}

Neuron &ANN::get_neuron_by_index(size_t const neuron_index_)
{
	return neurons[neuron_index_];
}

void ANN::declare_as_input(size_t const neuron_index_)
{
	if (!neurons[neuron_index_].is_input()) {
		neurons[neuron_index_].set_as_input();
		neurons[neuron_index_].set_output_function(TransferFunctions::identity);
		input_count++;
	}
}

void ANN::declare_as_output(size_t const neuron_index_)
{
	if (!neurons[neuron_index_].is_output()) {
		neurons[neuron_index_].set_as_output();
		neurons[neuron_index_].set_output_function(TransferFunctions::identity);
		output_count++;
		prev_output.resize(output_count);
	}
}

void ANN::set_neuron_input_function(size_t const neuron_index_, std::function<double(double, double)> input_function_)
{
	neurons[neuron_index_].set_input_function(input_function_);
}

void ANN::set_neuron_output_function(size_t const neuron_index_, std::function<double(double)> output_function_)
{
	neurons[neuron_index_].set_output_function(output_function_);
}

void ANN::train()
{

}
