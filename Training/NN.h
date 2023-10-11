///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <random>
#include <span>
#include "StackPoolAllocator.h"
#include "DualNumber.h"

// Note: using std::vector instead of std::array because using array made storing a neural net
// and gradients on the stack be in danger of running out of stack space, especially if layer
// sizes are changed for experimentation.  We lose compile time size checking though.
template <size_t NumInputNeurons, size_t NumHiddenNeurons, size_t NumOutputNeurons>
class NeuralNetwork
{
public:
	static const size_t c_numInputNeurons = NumInputNeurons;
	static const size_t c_numHiddenNeurons = NumHiddenNeurons;
	static const size_t c_numOutputNeurons = NumOutputNeurons;

	// There is a weight for each neuron in the previous layer, to each neuron in the current layer.
	// There is also one extra weight per neuron in each layer, for the bias term.
	// The activation of the previous layer will include an extra 1.0 for that bias term.
	static const size_t c_numHiddenWeights = (c_numInputNeurons + 1) * c_numHiddenNeurons;
	static const size_t c_numOutputWeights = (c_numHiddenNeurons + 1) * c_numOutputNeurons;
	static const size_t c_numWeights = c_numHiddenWeights + c_numOutputWeights;

	// initialize weights and biases to a gaussian distribution random number with mean 0, stddev 1.0
	NeuralNetwork(std::mt19937& rng)
	{
		std::normal_distribution<float> dist(0.0f, 1.0f);
		m_weights.resize(c_numWeights);
		for (float& f : m_weights)
			f = dist(rng);
	}

	template <typename T>
	std::span<const T, c_numWeights> GetWeights(StackPoolAllocator<T>& allocator) const;

	template <>
	std::span<const float, c_numWeights> GetWeights(StackPoolAllocator<float>& allocator) const
	{
		return std::span<const float, c_numWeights>{ m_weights.data(), c_numWeights };
	}

	template <>
	std::span<const DualNumber, c_numWeights> GetWeights(StackPoolAllocator<DualNumber>& allocator) const
	{
		// allocate dual numbers for the weights
		auto ret = allocator.Allocate<c_numWeights, false>();

		// Set their real and dual parts
		for (int i = 0; i < c_numWeights; ++i)
		{
			ret[i].Reset();
			ret[i].m_real = m_weights[i];
			ret[i].SetDualValue(i, 1.0f);
		}
		return ret;
	}

	// Returns the gradient, using backpropagation
	std::span<const float, c_numWeights> ForwardPassAndBackprop(std::span<const float, c_numInputNeurons + 1> input, int label) const
	{
		// We use a thread local stack allocator to get rid of allocation cost of local arrays.
		thread_local StackPoolAllocator<float> allocator(
			c_numHiddenNeurons + 1 +					// hiddenLayerActivations
			c_numOutputNeurons + 1 +					// outputLayerActivations
			c_numOutputNeurons +						// OutputLayer_deltaCost_deltaZ
			c_numHiddenNeurons * c_numOutputNeurons +	// OutputLayer_deltaCost_deltaWeight
			c_numHiddenNeurons +						// HiddenLayer_deltaCost_deltaZ
			c_numInputNeurons * c_numHiddenNeurons +    // HiddenLayer_deltaCost_deltaWeight
			c_numWeights                                // Gradient
		);
		allocator.Reset();


		// Evaluate the hidden layer
		auto hiddenWeights = std::span<const float, c_numHiddenWeights>{ &m_weights[0], c_numHiddenWeights };
		auto hiddenLayerActivations = EvaluateLayer(input, hiddenWeights, allocator);

		// Evaluate the output layer
		auto outputWeights = std::span<const float, c_numOutputWeights>{ &m_weights[c_numHiddenWeights], c_numOutputWeights };
		auto outputLayerActivations = EvaluateLayer(hiddenLayerActivations, outputWeights, allocator);

		// Do backpropagation
		{
			// The cost function of the total network that we want to minimize is the sum of the cost function of each output neuron.
			// 
			// The cost function of a single neuron is going to be 1/2 (desiredOutput - output)^2.
			// The 1/2 is there so that the derivative of the cost function (deltaCost/deltaOutput) is output - desiredOutput.
			// 
			// In the below:
			//  * Z is the output of that neuron before the activation function (the sum of the weighted inputs).
			//  * O ("oh") is Z put through the activation function, and is the neuron output value.


			// Output Layer Part 1
			// 
			// Calculate deltaCost/deltaZ for each output neuron.
			// This is also deltaCost/deltaBias since changing the bias changes Z directly, 1:1.
			//
			// deltaCost/deltaZ = deltaCost/deltaO * deltaO/deltaZ
			//
			// deltaCost/deltaO = O - desiredOutput
			// 
			// deltaO/deltaZ = O * (1 - O)
			//
			auto OutputLayer_deltaCost_deltaZ = allocator.Allocate<c_numOutputNeurons, false>();
			for (int outputNeuronIndex = 0; outputNeuronIndex < c_numOutputNeurons; ++outputNeuronIndex)
			{
				float desiredOutput = (outputNeuronIndex == label) ? 1.0f : 0.0f;
				float deltaCost_deltaO = outputLayerActivations[outputNeuronIndex] - desiredOutput;
				float deltaO_deltaZ = outputLayerActivations[outputNeuronIndex] * (1.0f - outputLayerActivations[outputNeuronIndex]);
				OutputLayer_deltaCost_deltaZ[outputNeuronIndex] = deltaCost_deltaO * deltaO_deltaZ;
			}

			// Output Layer Part 2
			//
			// Calculate deltaCost/deltaWeight for each weight going into each output neuron
			// 
			// deltaCost/deltaWeight = deltaCost/deltaZ * deltaZ/deltaWeight
			//
			// deltaZ/deltaWeight is the hidden layer activation that goes with each weight, since the weights and activations are multiplied together before being summed.
			// 
			// deltaCost/deltaWeight = deltaCost/deltaZ * hiddenLayerActivation
			//
			auto OutputLayer_deltaCost_deltaWeight = allocator.Allocate<c_numHiddenNeurons * c_numOutputNeurons, false>();
			for (int outputNeuronIndex = 0; outputNeuronIndex < c_numOutputNeurons; ++outputNeuronIndex)
			{
				for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < c_numHiddenNeurons; ++hiddenNeuronIndex)
				{
					OutputLayer_deltaCost_deltaWeight[outputNeuronIndex * c_numHiddenNeurons + hiddenNeuronIndex] = OutputLayer_deltaCost_deltaZ[outputNeuronIndex] * hiddenLayerActivations[hiddenNeuronIndex];
				}
			}

			// Hidden Layer Part 1
			// 
			// Calculate deltaCost/deltaZ for each hidden neuron.
			// This is also deltaCost/deltaBias since changing the bias changes Z directly, 1:1.
			// 
			// Each hidden layer neuron contributes to the error of each output neuron, so we need to sum them up the error from all of those paths.
			// 
			// Each path has a deltaCost/deltaO of deltaCost/deltaOutputZ * deltaOutputZ/deltaO.
			// 
			// Getting the deltaCost/deltaO of the entire neuron means summing up each path.
			// 
			// deltaCost/deltaO = Sum( deltaCost/deltaOutputZ * deltaOutputZ/deltaO )
			// 
			// deltaCost/deltaOutputZ is already calculated as OutputLayer_deltaCost_deltaZ.
			// 
			// deltaOutputZ/deltaO is the value of the weight connecting the hidden and output neuron, the output is multiplied by that weight for that output neuron.
			//
			auto HiddenLayer_deltaCost_deltaZ = allocator.Allocate<c_numHiddenNeurons, false>();
			for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < c_numHiddenNeurons; ++hiddenNeuronIndex)
			{
				float deltaCost_deltaO = 0.0f;
				for (int outputNeuronIndex = 0; outputNeuronIndex < c_numOutputNeurons; ++outputNeuronIndex)
					deltaCost_deltaO += OutputLayer_deltaCost_deltaZ[outputNeuronIndex] * m_weights[c_numHiddenWeights + outputNeuronIndex * (c_numHiddenNeurons + 1) + hiddenNeuronIndex];
				float deltaO_deltaZ = hiddenLayerActivations[hiddenNeuronIndex] * (1.0f - hiddenLayerActivations[hiddenNeuronIndex]);
				HiddenLayer_deltaCost_deltaZ[hiddenNeuronIndex] = deltaCost_deltaO * deltaO_deltaZ;
			}

			// Hidden Layer Part 2
			// 
			// Calculate deltaCost/deltaWeight for each weight going into the hidden neuron
			// 
			// deltaCost/deltaWeight = deltaCost/deltaZ * deltaZ/deltaWeight
			// 
			// deltaZ/deltaWeight is the input layer value that goes with each weight, since the weights and inputs are multiplied together before being summed.
			// 
			// deltaCost/deltaWeight = deltaCost/deltaBias * input
			//
			auto HiddenLayer_deltaCost_deltaWeight = allocator.Allocate<c_numInputNeurons * c_numHiddenNeurons, false>();
			for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < c_numHiddenNeurons; ++hiddenNeuronIndex)
			{
				for (int inputNeuronIndex = 0; inputNeuronIndex < c_numInputNeurons; ++inputNeuronIndex)
				{
					HiddenLayer_deltaCost_deltaWeight[hiddenNeuronIndex * c_numInputNeurons + inputNeuronIndex] = HiddenLayer_deltaCost_deltaZ[hiddenNeuronIndex] * input[inputNeuronIndex];
				}
			}

			// Copy our derivatives into the proper locations of the gradient array
			// 
			// This could be improved by having the above work in the larger gradient vector, instead of temporary arrays, but not doing that because
			// it would complicate the code, and these copies aren't showing up in profiling.
			auto gradient = allocator.Allocate<c_numWeights, false>();
			int outputIndex = 0;
			for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < c_numHiddenNeurons; ++hiddenNeuronIndex)
			{
				memcpy(&gradient[outputIndex], &HiddenLayer_deltaCost_deltaWeight[hiddenNeuronIndex * c_numInputNeurons], sizeof(float) * c_numInputNeurons);
				outputIndex += c_numInputNeurons;

				gradient[outputIndex] = HiddenLayer_deltaCost_deltaZ[hiddenNeuronIndex];
				outputIndex++;
			}

			for (int outputNeuronIndex = 0; outputNeuronIndex < c_numOutputNeurons; ++outputNeuronIndex)
			{
				memcpy(&gradient[outputIndex], &OutputLayer_deltaCost_deltaWeight[outputNeuronIndex * c_numHiddenNeurons], sizeof(float) * c_numHiddenNeurons);
				outputIndex += c_numHiddenNeurons;

				gradient[outputIndex] = OutputLayer_deltaCost_deltaZ[outputNeuronIndex];
				outputIndex++;
			}

			return gradient;
		}
	}

	// Templated so it can take either floats or dual numbers
	template <typename T>
	std::span<const T, c_numOutputNeurons> Evaluate(std::span<const float, c_numInputNeurons + 1> input) const
	{
		// We use a thread local stack allocator to get rid of allocation cost of local arrays.
		thread_local StackPoolAllocator<T> allocator(c_numHiddenNeurons + 1 + c_numOutputNeurons + 1 + c_numWeights);
		allocator.Reset();

		// This is where weights get converted to dual numbers, if needed
		auto weights = GetWeights<T>(allocator);

		// Evaluate the hidden layer
		auto hiddenWeights = std::span<const T, c_numHiddenWeights>{ &weights[0], c_numHiddenWeights };
		auto hiddenLayer = EvaluateLayer(input, hiddenWeights, allocator);

		// Evaluate the output layer
		auto outputWeights = std::span<const T, c_numOutputWeights>{ &weights[c_numHiddenWeights], c_numOutputWeights };
		auto outputLayerActivations = EvaluateLayer(hiddenLayer, outputWeights, allocator);

		// Remove the extra 1.0 at the end, since we are done and there is no next layer with a bias term
		return std::span<const T, c_numOutputNeurons>{outputLayerActivations.first(c_numOutputNeurons).data(), c_numOutputNeurons};
	}

	int EvaluateOneHot(std::span<const float, c_numInputNeurons + 1> input) const
	{
		// Evaluate the network
		auto outputLayerActivations = Evaluate<float>(input);

		// return the index of the most activated output neuron
		int bestNeuron = 0;
		float bestNeuronActivation = outputLayerActivations[0];
		for (int i = 1; i < c_numOutputNeurons; ++i)
		{
			if (outputLayerActivations[i] > bestNeuronActivation)
			{
				bestNeuron = i;
				bestNeuronActivation = outputLayerActivations[i];
			}
		}
		return bestNeuron;
	}

	// Cost is mean squared error
	// Templated so it can take either floats or dual numbers
	template <typename T>
	T EvaluateOneHotCost(std::span<const float, c_numInputNeurons + 1> input, int expectedOutput) const
	{
		T ret = {};

		// Evaluate the network
		auto outputLayerActivations = Evaluate<T>(input);

		// Calculate and return mean squared error
		for (size_t i = 0; i < c_numOutputNeurons; ++i)
		{
			float target = (i == expectedOutput) ? 1.0f : 0.0f;
			T error = target - outputLayerActivations[i];

			// This is a way of doing "online averaging" that keeps numbers similar sized to avoid floating point precision issues
			ret = Lerp(ret, error * error, 1.0f / float(i + 1));
		}
		return ret;
	}

	void UpdateWeights(const std::vector<float>& gradient, float learningRate)
	{
		// validate input
		if (gradient.size() != c_numWeights)
		{
			printf("ERROR: " __FUNCTION__ "(): gradient is the wrong size.\n");
			return;
		}

		// apply update
		for (size_t i = 0; i < c_numWeights; ++i)
			m_weights[i] -= gradient[i] * learningRate;
	}

	float& GetWeight(size_t index)
	{
		return m_weights[index];
	}

private:

	template <typename T, typename U, size_t NUM_ACTIVATIONS, size_t NUM_WEIGHTS>
	inline std::span<const T, NUM_WEIGHTS / NUM_ACTIVATIONS + 1> EvaluateLayer(const std::span<const U, NUM_ACTIVATIONS>& activations, const std::span<const T, NUM_WEIGHTS>& weights, StackPoolAllocator<T>& allocator) const
	{
		constexpr const size_t c_numActivations = NUM_ACTIVATIONS;
		constexpr const size_t c_neurons = NUM_WEIGHTS / NUM_ACTIVATIONS;

		// Do a vector by matrix multiply.
		// Activations is the vector, weights is the matrix
		auto ret = allocator.Allocate<c_neurons + 1, false>();
		for (size_t i = 0; i < c_neurons; ++i)
		{
			ret[i] = DotProduct(&weights[i * c_numActivations], &activations[0], c_numActivations);
			ret[i] = ActivationFunction(ret[i]);
		}

		// An extra activation value for the bias term of the next layer
		ret[c_neurons] = T(1.0f);
		return ret;
	}

	template <typename T>
	inline static T ActivationFunction(T x)
	{
		return 1.0f / (1.0f + std::exp(-x));
	}

	template <typename T, typename U>
	inline static T DotProduct(const T* A, const U* B, size_t N)
	{
		T ret = T(0.0f);
		for (size_t i = 0; i < N; ++i)
			ret += A[i] * B[i];
		return ret;
	}

	template <typename T>
	inline static T Lerp(const T& A, const T& B, float t)
	{
		return A * (1.0f - t) + B * t;
	}

	std::vector<float> m_weights;
};
