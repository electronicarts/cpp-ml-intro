///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

Buffer<float> NNWeights : register(t0);
Buffer<float> HiddenLayerActivations : register(t1);
RWBuffer<float> OutputLayerActivations : register(u0);


[numthreads(64, 1, 1)]
void OutputLayer(uint3 DTid : SV_DispatchThreadID)
{
    int outputNeuronIndex = DTid.x;

    // Calculate where the weights begin and end for this neuron.
    // There is an extra weight for the bias
    int weightsBeginIndex = (23550) + outputNeuronIndex * ((30) + 1);

    float output = NNWeights[weightsBeginIndex + int(30)]; // bias
    for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < int(30); ++hiddenNeuronIndex)
        output += HiddenLayerActivations[hiddenNeuronIndex] * NNWeights[weightsBeginIndex + hiddenNeuronIndex];

    // activation function
    OutputLayerActivations[outputNeuronIndex] = 1.0f / (1.0f + exp(-output));
}
