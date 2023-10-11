///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#include "Settings.h"
#include "DataSet.h"

std::span<const float, TNeuralNetwork::c_numWeights> GetGradient_DualNumbers(TNeuralNetwork& neuralNet, const DataItem& dataItem)
{
	static std::vector<float> gradient(TNeuralNetwork::c_numWeights);

	// Evaluate it and get the cost as a dual number
	DualNumber cost = neuralNet.EvaluateOneHotCost<DualNumber>(dataItem.image, dataItem.label);

	// extract the gradient
	for (int i = 0; i < gradient.size(); ++i)
		gradient[i] = -cost.GetDualValue(i);

	return std::span<const float, TNeuralNetwork::c_numWeights>{ gradient.data(), TNeuralNetwork::c_numWeights };
}
