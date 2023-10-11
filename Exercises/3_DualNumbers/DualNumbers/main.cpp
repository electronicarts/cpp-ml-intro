#include <stdio.h>
#include <random>

static const int c_numSteps = 50;
static const bool c_deterministic = false; // Set to true to get the same random numbers every run. useful for debugging
static const float c_gradientStepSize = 0.1f;

struct CDualNumber
{
	CDualNumber()
	{

	}

	// Conversion from floating point constant to dual number
	CDualNumber(float f)
		: m_real(f)
		, m_dual(0.0f)
	{
	}
	
	CDualNumber(float real, float dual)
		: m_real(real)
		, m_dual(dual)
	{
	}

	float m_real = 0.0f;  // The value
	float m_dual = 0.0f;  // The derivative. Constant values will always have a derivative / dual value of 0.
	float m_dualX = 0.0f;
	float m_dualY = 0.0f;

	// TODO: implement these operators to make the program run

	CDualNumber operator+(const CDualNumber& d)
	{

	}

	CDualNumber operator-(const CDualNumber& d)
	{

	}

	CDualNumber operator*(const CDualNumber& d)
	{

	}
};

template <typename T>
T F(T x)
{
	// y = (x+1)^2-2
	// The minimum is at x=-1, with a y value of -2
	return (x + 1.0f) * (x + 1.0f) - 2.0f;
}

float FDerivative(float x)
{
	// Calculate the derivative by making a dual number and putting it through the F() function
	CDualNumber dualX(x, 1.0f);
	return F(dualX).m_dual;
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
