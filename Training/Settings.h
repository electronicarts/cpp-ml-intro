///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "NN.h"

#define TRAIN_FORWARD_DIFF() false
#define TRAIN_CENTRAL_DIFF() false
#define TRAIN_DUAL_NUMBERS() false
#define TRAIN_BACKPROP() true

#define DETERMINISTIC() false
#define MULTI_THREADED() true

static const size_t c_imageDims = 28;

const size_t c_trainingEpochs = 30;	// How many times we go through all of the training data.
const size_t c_miniBatchSize = 10;	// How many items of the training data we should train against, at a time.
const float c_learningRate = 3.0f;	// How fast should we travel down the gradient.

const float c_finiteDifferencesEpsilon = 0.01f; // The epsilon used in finite differences
const size_t c_finiteDifferencesThreadSize = 100; // How many weights should each thread handle when doing finite differences?

// Our neural network has:
//  * 784 input neurons.  1 input neuron for each pixel.
//  * 30 hidden neurons.  To help find how to match input to output.
//  * 10 output neurons.  To specify the digit 0 to 9.
using TNeuralNetwork = NeuralNetwork<c_imageDims* c_imageDims, 30, 10>;

struct DataItem;

std::span<const float, TNeuralNetwork::c_numWeights> GetGradient_FiniteDifferences_Central(TNeuralNetwork& neuralNet, const DataItem& dataItem);
std::span<const float, TNeuralNetwork::c_numWeights> GetGradient_FiniteDifferences_Forward(TNeuralNetwork& neuralNet, const DataItem& dataItem);
std::span<const float, TNeuralNetwork::c_numWeights> GetGradient_DualNumbers(TNeuralNetwork& neuralNet, const DataItem& dataItem);
std::span<const float, TNeuralNetwork::c_numWeights> GetGradient_Backprop(TNeuralNetwork& neuralNet, const DataItem& dataItem);