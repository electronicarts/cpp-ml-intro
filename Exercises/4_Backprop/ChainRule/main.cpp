#include <stdio.h>
#include <random>

static const int c_numSteps = 50;
static const bool c_deterministic = false; // Set to true to get the same random numbers every run. useful for debugging
static const float c_gradientStepSize = 0.1f;

float G(float x)
{
	return x * x;
}

float GDerivative(float x)
{
	return 2.0f * x;
}

float H(float x)
{
	return 3.0f * x + 5.0f;
}

float HDerivative(float x)
{
	return 3.0f;
}

float I(float x)
{
	return x * 0.9f + 5.0f;
}

float IDerivative(float x)
{
	return 0.9f;
}

float F(float x)
{
	return G(H(I(x)));
}

float FDerivative(float x)
{
	// TODO: calculate the derivative of F using the chain rule
}

std::mt19937 GetRNG()
{
	if (c_deterministic)
	{
		std::mt19937 rng;
		return rng;
	}
	else
	{
		std::random_device rd;
		std::mt19937 rng(rd());
		return rng;
	}
}

int main(int argc, char** argv)
{
	std::mt19937 rng = GetRNG();
	std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

	float x = dist(rng);

	int lastPercent = -1;
	for (int i = 0; i < c_numSteps; ++i)
	{
		float derivative = FDerivative(x);

		int percent = int(100.0f * float(i) / float(c_numSteps - 1));
		if (percent != lastPercent)
		{
			printf("%3i%%: x = %0.2f, y = %0.2f, y' = %0.2f\n", percent, x, F(x), derivative);
			lastPercent = percent;
		}

		x -= derivative * c_gradientStepSize;
	}

	return 0;
}
