///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>
#include <array>
#include "Settings.h"
#include <span>

struct DataItem
{
	int label;
	float image[c_imageDims * c_imageDims + 1]; // We have an extra 1.0 value for the input layer bias term
};

typedef std::vector<DataItem> DataSet;

void ExtractMNISTData(DataSet& trainingData, DataSet& testingData);