///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#define _CRT_SECURE_NO_WARNINGS

#include "DataSet.h"

#include <vector>
#include <stdint.h>
#include <direct.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Demo/mnist/DX12Utils/stb/stb_image_write.h"

typedef std::vector<uint8_t> Bytes;

inline uint32_t EndianSwap(uint32_t a)
{
	return (a << 24) | ((a << 8) & 0x00ff0000) |
		((a >> 8) & 0x0000ff00) | (a >> 24);
}

Bytes LoadFileIntoMemory(const char* fileName)
{
	Bytes ret;

	FILE* file = nullptr;
	fopen_s(&file, fileName, "rb");
	if (!file)
		return ret;

	fseek(file, 0, SEEK_END);
	ret.resize(ftell(file), 0);
	fseek(file, 0, SEEK_SET);

	fread(ret.data(), 1, ret.size(), file);

	fclose(file);
	return ret;
}

struct DataFiles
{
	Bytes labelFile;
	uint32_t labelCount = 0;
	uint8_t* labels = nullptr;

	Bytes imageFile;
	uint32_t imageCount = 0;
	uint8_t* pixels = nullptr;
};

DataFiles LoadLabelAndDataFile(const char* labelFileName, const char* imageFileName)
{
	DataFiles ret;

	// Load and verify label data.
	ret.labelFile = LoadFileIntoMemory(labelFileName);
	uint32_t* labelData = (uint32_t*)ret.labelFile.data();
	if (labelData[0] == 0x01080000)
	{
		labelData[0] = EndianSwap(labelData[0]);
		labelData[1] = EndianSwap(labelData[1]);
	}
	if (labelData[0] != 2049)
		return ret;
	ret.labelCount = labelData[1];
	ret.labels = (uint8_t*)(&labelData[2]);

	// load and verify image data
	ret.imageFile = LoadFileIntoMemory(imageFileName);
	uint32_t* imageData = (uint32_t*)ret.imageFile.data();
	if (imageData[0] == 0x03080000)
	{
		imageData[0] = EndianSwap(imageData[0]);
		imageData[1] = EndianSwap(imageData[1]);
		imageData[2] = EndianSwap(imageData[2]);
		imageData[3] = EndianSwap(imageData[3]);
	}
	if (imageData[0] != 2051 || imageData[2] != 28 || imageData[3] != 28) // images are 28x28
		return ret;
	ret.imageCount = imageData[1];
	ret.pixels = (uint8_t*)(&imageData[4]);

	return ret;
}

void Convert(const DataFiles& dataFiles, const char* outDir)
{
	printf("%s...\n", outDir);

	// Save out images
	int lastPercent = -1;
	std::vector<int> fileCounts(10, 0);
	uint8_t* pixels = dataFiles.pixels;
	for (uint32_t i = 0; i < dataFiles.imageCount; ++i)
	{
		int percent = int(100.0f * float(i) / float(dataFiles.imageCount - 1));
		if (percent != lastPercent)
		{
			lastPercent = percent;
			printf("\r%i%%", percent);
		}

		char fileName[1024];
		sprintf(fileName, "%s%i_%i.png", outDir, (int)dataFiles.labels[i], fileCounts[dataFiles.labels[i]]);
		stbi_write_png(fileName, 28, 28, 1, pixels, 0);
		pixels += 28 * 28;
		fileCounts[dataFiles.labels[i]]++;
	}
	printf("\r100%%\n");
}

void ExtractMNISTData(DataSet& trainingData, DataSet& testingData)
{
	DataFiles training = LoadLabelAndDataFile("../Data/mnist/train-labels.idx1-ubyte", "../Data/mnist/train-images.idx3-ubyte");
	DataFiles testing = LoadLabelAndDataFile("../Data/mnist/t10k-labels.idx1-ubyte", "../Data/mnist/t10k-images.idx3-ubyte");

	// extract the mnist data to PNGs, but only if it isn't already extracted
	{
		FILE* file = nullptr;
		fopen_s(&file, "../Data/Testing/0_0.png", "rb");
		if (file)
		{
			fclose(file);
		}
		else
		{
			// Extract the data
			_mkdir("../Data/Training/");
			Convert(training, "../Data/Training/");

			_mkdir("../Data/Testing/");
			Convert(testing, "../Data/Testing/");
		}
	}

	// Fill out training data
	trainingData.resize(training.imageCount);
	for (int imageIndex = 0; imageIndex < (int)training.imageCount; ++imageIndex)
	{
		DataItem& item = trainingData[imageIndex];
		item.label = training.labels[imageIndex];
		for (int pixelIndex = 0; pixelIndex < c_imageDims * c_imageDims; ++pixelIndex)
			item.image[pixelIndex] = float(training.pixels[imageIndex * c_imageDims * c_imageDims + pixelIndex]) / 255.0f;
		item.image[c_imageDims * c_imageDims] = 1.0f;
	}

	// fill out testing data
	testingData.resize(testing.imageCount);
	for (int imageIndex = 0; imageIndex < (int)testing.imageCount; ++imageIndex)
	{
		DataItem& item = testingData[imageIndex];
		item.label = testing.labels[imageIndex];
		for (int pixelIndex = 0; pixelIndex < c_imageDims * c_imageDims; ++pixelIndex)
			item.image[pixelIndex] = float(testing.pixels[imageIndex * c_imageDims * c_imageDims + pixelIndex]) / 255.0f;
		item.image[c_imageDims * c_imageDims] = 1.0f;
	}
}