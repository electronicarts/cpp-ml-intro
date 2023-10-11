///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#include "Settings.h"
#include "DataSet.h"

std::span<const float, TNeuralNetwork::c_numWeights> GetGradient_Backprop(TNeuralNetwork& neuralNet, const DataItem& dataItem)
{
	return neuralNet.ForwardPassAndBackprop(dataItem.image, dataItem.label);
}