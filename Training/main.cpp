///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <numeric>
#include <chrono>
#include <direct.h>

#include "DataSet.h"

#include "Settings.h"

std::mt19937 GetRNG()
{
#if DETERMINISTIC()
	std::mt19937 rng;
#else
	std::random_device rd;
	std::mt19937 rng(rd());
#endif
	return rng;
}

float EvaluateNetworkQuality(const TNeuralNetwork& nn, const DataSet& testingData)
{
	int correct = 0;
	for (const DataItem& item : testingData)
	{
		int predictedLabel = nn.EvaluateOneHot(item.image);
		if (predictedLabel == item.label)
			correct++;
	}

	float accuracyPercent = 100.0f * float(correct) / float(testingData.size());

	printf("Accuracy: %0.2f%% (%i incorrect)\n", accuracyPercent, int(testingData.size() - correct));
	return accuracyPercent;
}

std::string MakeDurationString(float durationInSeconds)
{
	std::string ret;

	static const float c_oneMinute = 60.0f;
	static const float c_oneHour = c_oneMinute * 60.0f;

	int hours = int(durationInSeconds / c_oneHour);
	durationInSeconds -= float(hours) * c_oneHour;

	int minutes = int(durationInSeconds / c_oneMinute);
	durationInSeconds -= float(minutes) * c_oneMinute;

	int seconds = int(durationInSeconds);

	char buffer[1024];
	if (hours < 10)
		sprintf_s(buffer, "0%i:", hours);
	else
		sprintf_s(buffer, "%i:", hours);
	ret = buffer;

	if (minutes < 10)
		sprintf_s(buffer, "0%i:", minutes);
	else
		sprintf_s(buffer, "%i:", minutes);
	ret += buffer;

	if (seconds < 10)
		sprintf_s(buffer, "0%i", seconds);
	else
		sprintf_s(buffer, "%i", seconds);
	ret += buffer;

	return ret;
}

template <typename LAMBDA>
void Train(const DataSet& trainingData, const DataSet& testingData, LAMBDA GetGradient, const char* name)
{
	// Remember when the training started so we can report the time duration later
	std::chrono::high_resolution_clock::time_point trainingStart = std::chrono::high_resolution_clock::now();

	std::mt19937 rng = GetRNG();

	TNeuralNetwork nn(rng);
	std::vector<float> gradientSum(TNeuralNetwork::c_numWeights);

	// Make a list of indices in our training data. We'll shuffle this each epoch and then train in that order
	std::vector<int> trainingOrder(trainingData.size());
	std::iota(trainingOrder.begin(), trainingOrder.end(), 0);

	// Each epoch is a training with the entire list of training data
	std::vector<float> epochAccuracy(c_trainingEpochs);
	for (size_t epoch = 0; epoch < c_trainingEpochs; ++epoch)
	{
		// Remember when the epoch started so we can report the time duration later
		std::chrono::high_resolution_clock::time_point epochStart = std::chrono::high_resolution_clock::now();

		// For reporting progress
		int lastPercent = -1;

		// randomize the order that we are going to use the training data in, for this epoch
		std::shuffle(trainingOrder.begin(), trainingOrder.end(), rng);

		// Do each mini batch
		size_t trainingIndex = 0;
		while (trainingIndex < trainingOrder.size())
		{
			// Get the summed gradient for a mini batch
			std::fill(gradientSum.begin(), gradientSum.end(), 0.0f);

			size_t trainingEndIndex = std::min(trainingIndex + c_miniBatchSize, trainingOrder.size());
			size_t trainingCount = trainingEndIndex - trainingIndex;
			while (trainingIndex < trainingEndIndex)
			{
				std::span < const float, TNeuralNetwork::c_numWeights> gradient = GetGradient(nn, trainingData[trainingOrder[trainingIndex]]);
				for (size_t index = 0; index < gradient.size(); ++index)
					gradientSum[index] += gradient[index];

				trainingIndex++;

				int percent = int(1000.0f * float(trainingIndex) / float(trainingData.size()));
				if (percent != lastPercent)
				{
					lastPercent = percent;
					printf("\r[Epoch %i/%i] %0.2f%%", (int)epoch + 1, (int)c_trainingEpochs, float(percent) / 10.0f);
				}
			}

			// Adjust the weights of the network by the gradient.
			// Divide the trainingCount to make it an average gradient though, and multiply by the learning rate
			nn.UpdateWeights(gradientSum, c_learningRate / float(trainingCount));
		}

		float epochDuration = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - epochStart).count();
		printf("\r[Epoch %i/%i] Duration: %s ", (int)epoch + 1, (int)c_trainingEpochs, MakeDurationString(epochDuration).c_str());
		epochAccuracy[epoch] = EvaluateNetworkQuality(nn, testingData);
	}

	float trainingDuration = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - trainingStart).count();
	printf("[Total] Duration %s ", MakeDurationString(trainingDuration).c_str());
	EvaluateNetworkQuality(nn, testingData);

	// save accuracy as csv
	{
		char fileName[256];
		sprintf_s(fileName, "out/%s_Accuracy.csv", name);

		FILE* file = nullptr;
		fopen_s(&file, fileName, "wb");

		fprintf(file, "\"Epoch\",\"%s\"\n", name);

		for (int i = 0; i < epochAccuracy.size(); ++i)
			fprintf(file, "\"%i\",\"%f\"\n", i + 1, epochAccuracy[i]);

		fclose(file);
	}

	// Save the weights as csv
	{
		char fileName[256];
		sprintf_s(fileName, "out/%s_Weights.csv", name);

		FILE* file = nullptr;
		fopen_s(&file, fileName, "wb");

		// write the hidden layer
		int weightIndex = 0;
		for (size_t hiddenNeuronIndex = 0; hiddenNeuronIndex < TNeuralNetwork::c_numHiddenNeurons; ++hiddenNeuronIndex)
		{
			for (size_t inputNeuronIndex = 0; inputNeuronIndex < TNeuralNetwork::c_numInputNeurons; ++inputNeuronIndex)
			{
				fprintf(file, "\"Input%i to Hidden%i Weight\",\"%f\"\n", (int)inputNeuronIndex, (int)hiddenNeuronIndex, nn.GetWeight(weightIndex));
				weightIndex++;
			}

			fprintf(file, "\"Hidden%i Bias\",\"%f\"\n", (int)hiddenNeuronIndex, nn.GetWeight(weightIndex));
			weightIndex++;
		}

		// write the output later
		for (size_t outputNeuronIndex = 0; outputNeuronIndex < TNeuralNetwork::c_numOutputNeurons; ++outputNeuronIndex)
		{
			for (size_t hiddenNeuronIndex = 0; hiddenNeuronIndex < TNeuralNetwork::c_numHiddenNeurons; ++hiddenNeuronIndex)
			{
				fprintf(file, "\"Hidden%i to Output%i Weight\",\"%f\"\n", (int)hiddenNeuronIndex, (int)outputNeuronIndex, nn.GetWeight(weightIndex));
				weightIndex++;
			}

			fprintf(file, "\"Output%i Bias\",\"%f\"\n", (int)outputNeuronIndex, nn.GetWeight(weightIndex));
			weightIndex++;
		}

		fclose(file);
	}

	// Save the weights as binary
	{
		char fileName[256];
		sprintf_s(fileName, "out/%s_Weights.bin", name);

		FILE* file = nullptr;
		fopen_s(&file, fileName, "wb");
		fwrite(&nn.GetWeight(0), sizeof(float), TNeuralNetwork::c_numWeights, file);
		fclose(file);
	}
}

int main(int argc, char** argv)
{
	_mkdir("out");

	// Make the MNIST data into .png files.
	// Not necessary, but it makes it easier to see what the training data looks like, having it on disk as pngs.
	printf("Extracting MNIST Data...\n");
	DataSet trainingData, testingData;
	ExtractMNISTData(trainingData, testingData);

	// You can uncomment this if you want to train / test on a random subset of data.
	/*
	std::shuffle(trainingData.begin(), trainingData.end(), GetRNG());
	std::shuffle(testingData.begin(), testingData.end(), GetRNG());
	trainingData.resize(100);
	testingData.resize(100);
	*/

	printf("MLP layers are: %i, %i, %i, for a total of %i weights to optimize.\n",
		(int)TNeuralNetwork::c_numInputNeurons,
		(int)TNeuralNetwork::c_numHiddenNeurons,
		(int)TNeuralNetwork::c_numOutputNeurons,
		(int)TNeuralNetwork::c_numWeights
	);

	#if TRAIN_FORWARD_DIFF()
		printf("\nTraining with Forward Differences...\n");
		Train(trainingData, testingData, GetGradient_FiniteDifferences_Forward, "ForwardDiff");
	#endif

	#if TRAIN_CENTRAL_DIFF()
		printf("\nTraining with Central Differences...\n");
		Train(trainingData, testingData, GetGradient_FiniteDifferences_Central, "CentralDiff");
	#endif

	#if TRAIN_DUAL_NUMBERS()
		printf("\nTraining with Dual Numbers...\n");
		Train(trainingData, testingData, GetGradient_DualNumbers, "DualNumbers.csv");
	#endif

	#if TRAIN_BACKPROP()
		printf("\nTraining with backprop...\n");
		Train(trainingData, testingData, GetGradient_Backprop, "Backprop");
	#endif

	return 0;
}
