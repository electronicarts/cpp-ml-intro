///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

Texture2D<float> NNInput : register(t0);
Buffer<float> NNWeights : register(t1);
RWBuffer<float> HiddenLayerActivations : register(u0);

#line 1


[numthreads(64, 1, 1)]
#line 3
void HiddenLayer(uint3 DTid : SV_DispatchThreadID)
{
    int hiddenNeuronIndex = DTid.x;

    // Calculate where the weights begin and end for this neuron.
    // There is an extra weight for the bias
    int weightsBeginIndex = hiddenNeuronIndex * ((784) + 1);

    const uint2 inputResolution = uint2(28, 28);

    float output = NNWeights[weightsBeginIndex + int(784)]; // bias
    for (int inputNeuronIndex = 0; inputNeuronIndex < int(784); ++inputNeuronIndex)
    {
        uint2 inputPixelPos = uint2(inputNeuronIndex % inputResolution.x, inputNeuronIndex / inputResolution.x );
        output += NNInput[inputPixelPos] * NNWeights[weightsBeginIndex + inputNeuronIndex];
    }

    // activation function
    HiddenLayerActivations[hiddenNeuronIndex] = 1.0f / (1.0f + exp(-output));
}