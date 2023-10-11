#include <stdio.h>
#include <random>

static const int c_numSteps = 50;
static const bool c_deterministic = false; // Set to true to get the same random numbers every run. useful for debugging
static const float c_gradientStepSize = 0.1f;

float F(float x)
{
	// y = (x+1)^2-2
	// The minimum is at x=-1, with a y value of -2
	return (x + 1.0f) * (x + 1.0f) - 2.0f;
}

float FDerivative(float x)
{
	// TODO: implement finite difference to return the derivative of function F(), at location x.
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