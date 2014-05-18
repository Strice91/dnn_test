#ifndef ANN_H
#define ANN_H

#include <vector>
#include "Neuron.h"

 class ANN
{
	public:
		explicit ANN(size_t const &neuron_count_);
		virtual ~ANN();

		void train();
		void connect_neurons(size_t const first_, size_t const second_, double const weight_ = 1.0);
		void disconnect_neurons(size_t const first_, size_t const second_);
		void init_weights_random();
		void init_weights_random(double const lower_bound_, double const upper_bound_);

		void set_weight_by_index(size_t const index_, double const &value_);
		void set_biases(std::vector<double> const &biases_);
		void set_weights(std::vector<std::vector<double>> const &weights_);
		void set_neuron_input_function(size_t const neuron_index_, std::function<double(double, double)> input_function_);
		void set_neuron_output_function(size_t const neuron_index_, std::function<double(double)> output_function_);

		void declare_as_input(size_t const neuron_index_);
		void declare_as_output(size_t const neuron_index_);
		void reset_internal_memory();

		void display_connections() const;
		void display_weights() const;
		void display_biases() const;

		double calculate_single_output(double const &input_) const;
		double calculate_single_output(std::vector<double> const &input_) const;
		std::vector<double> calculate_output(double const &input_) const;
		std::vector<double> calculate_output(std::vector<double> const &input_) const;

		size_t get_neuron_count() const;
		size_t get_input_count() const;
		size_t get_output_count() const;
		size_t get_connection_count() const;
		Neuron &get_neuron_by_index(size_t const neuron_index_);

		struct TransferFunctions
		{
			public:
				static double sigmoid(double d);
				static double tangent_hyperbolic(double d);
				static double identity(double d);
				static double heaviside(double d);
				static double ReLU(double d);

			private:
				TransferFunctions();
				~TransferFunctions();
		};

	private:
		struct ANNConnection
		{
			double weight;
			bool exists;
			bool recurrent;
		};
		std::vector<std::vector<ANNConnection>> adjactancy_matrix;


		std::vector<Neuron> neurons;
		std::vector<std::vector<double>> weights;
		std::vector<double> biases;
		std::vector<std::vector<bool>> connections;
		std::vector<double> prev_output;
		mutable std::vector<size_t> sorted_indices;
		mutable bool sort_required;
		void topological_sort() const;
		bool compare_topology(size_t const &first, size_t const &second) const;
		size_t neuron_count;
		size_t input_count;
		size_t output_count;
		size_t connection_count;
};

#endif
