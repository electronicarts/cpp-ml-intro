///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#include "Settings.h"
#include "DataSet.h"

#include <omp.h>
#include <thread>
#include <atomic>

std::span<const float, TNeuralNetwork::c_numWeights> GetGradient_FiniteDifferences_Forward(TNeuralNetwork& neuralNet, const DataItem& dataItem)
{
	static std::vector<float> gradient(TNeuralNetwork::c_numWeights);

	// Evaluate the network with no changes and calculate the cost function.
	float baseCost = neuralNet.EvaluateOneHotCost<float>(dataItem.image, dataItem.label);

	// We are going to go wide on threads, so we need a copy of the neural network for each thread!
	std::vector<TNeuralNetwork> NNCopies(std::thread::hardware_concurrency(), neuralNet);

	// Evaluate the network with small changes to each parameter individually.
	// The change in cost determines the gradient.
	size_t dispatchCount = (TNeuralNetwork::c_numWeights + c_finiteDifferencesThreadSize - 1) / c_finiteDifferencesThreadSize;
	#if MULTI_THREADED()
	#pragma omp parallel for
	#endif
	for (int i = 0; i < dispatchCount; ++i)
	{
		// Since we don't evaluate the derivatives for input weights where the input is zero,
		// it makes the workload uneven.  I tried using an atomic int to get the next dispatch index
		// instead of using i, but it slowed things down by like 1/3.
		int dispatchIndex = i;
		size_t weightIndexStart = dispatchIndex * c_finiteDifferencesThreadSize;
		size_t weightIndexEnd = std::min(weightIndexStart + c_finiteDifferencesThreadSize, TNeuralNetwork::c_numWeights);

		for (size_t weightIndex = weightIndexStart; weightIndex < weightIndexEnd; ++weightIndex)
		{
			// if this weight is for an input neuron that is 0 in this data set, we already know that the derivative is 0!
			// This makes the time drop to about 25% of the time it takes without this.
			if (weightIndex < TNeuralNetwork::c_numHiddenWeights)
			{
				size_t inputIndex = weightIndex % (TNeuralNetwork::c_numInputNeurons + 1);
				if (inputIndex < TNeuralNetwork::c_numInputNeurons && dataItem.image[inputIndex] == 0.0f)
				{
					gradient[weightIndex] = 0.0f;
					continue;
				}
			}

			TNeuralNetwork& nn = NNCopies[omp_get_thread_num()];

			// Adjust the current weight by a small amount, evaluate the neural network, then put the weight back
			float& weight = nn.GetWeight(weightIndex);
			float oldValue = weight;
			weight += c_finiteDifferencesEpsilon;
			float cost = nn.EvaluateOneHotCost<float>(dataItem.image, dataItem.label);
			weight = oldValue;

			// set the partial derivative for this weight
			gradient[weightIndex] = (cost - baseCost) / c_finiteDifferencesEpsilon;
		}
	}

	return std::span<const float, TNeuralNetwork::c_numWeights>{ gradient.data(), TNeuralNetwork::c_numWeights };
}

std::span<const float, TNeuralNetwork::c_numWeights> GetGradient_FiniteDifferences_Central(TNeuralNetwork& neuralNet, const DataItem& dataItem)
{
	static std::vector<float> gradient(TNeuralNetwork::c_numWeights);

	// We are going to go wide on threads, so we need a copy of the neural network for each thread!
	std::vector<TNeuralNetwork> NNCopies(std::thread::hardware_concurrency(), neuralNet);

	// Evaluate the network with small changes to each parameter individually.
	// The change in cost determines the gradient.
	size_t dispatchCount = (TNeuralNetwork::c_numWeights + c_finiteDifferencesThreadSize - 1) / c_finiteDifferencesThreadSize;
	#if MULTI_THREADED()
	#pragma omp parallel for
	#endif
	for (int i = 0; i < dispatchCount; ++i)
	{
		size_t weightIndexStart = i * c_finiteDifferencesThreadSize;
		size_t weightIndexEnd = std::min(weightIndexStart + c_finiteDifferencesThreadSize, TNeuralNetwork::c_numWeights);

		for (size_t weightIndex = weightIndexStart; weightIndex < weightIndexEnd; ++weightIndex)
		{
			// if this weight is for an input neuron that is 0 in this data set, we already know that the derivative is 0!
			// This makes the time drop to about 25% of the time it takes without this.
			if (weightIndex < TNeuralNetwork::c_numHiddenWeights)
			{
				size_t inputIndex = weightIndex % (TNeuralNetwork::c_numInputNeurons + 1);
				if (inputIndex < TNeuralNetwork::c_numInputNeurons && dataItem.image[inputIndex] == 0.0f)
				{
					gradient[weightIndex] = 0.0f;
					continue;
				}
			}

			TNeuralNetwork& nn = NNCopies[omp_get_thread_num()];

			// Adjust the current weight by a small amount, evaluate the neural network, then put the weight back
			float& weight = nn.GetWeight(weightIndex);
			float oldValue = weight;
			weight -= c_finiteDifferencesEpsilon;
			float cost1 = nn.EvaluateOneHotCost<float>(dataItem.image, dataItem.label);

			weight = oldValue + c_finiteDifferencesEpsilon;
			float cost2 = nn.EvaluateOneHotCost<float>(dataItem.image, dataItem.label);

			weight = oldValue;

			// set the partial derivative for this weight
			gradient[weightIndex] = (cost2 - cost1) / (2.0f * c_finiteDifferencesEpsilon);
		}
	}

	return std::span<const float, TNeuralNetwork::c_numWeights>{ gradient.data(), TNeuralNetwork::c_numWeights };
}
