///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>

#define SHRINK_DUALS() true
#define SHRINK_DUALS_ZERO_THRESHOLD() 0.0001f // This can be set to 0.0f

struct DualNumber
{
	DualNumber()
	{
		m_real = 0.0f;
		m_dualIndexMin = INT_MAX;
		m_dualIndexMax = INT_MIN;
	}

	explicit DualNumber(float f)
	{
		m_real = f;
		m_dualIndexMin = INT_MAX;
		m_dualIndexMax = INT_MIN;
	}

	void Reset()
	{
		m_real = 0.0f;
		m_dualIndexMin = INT_MAX;
		m_dualIndexMax = INT_MIN;
		m_dual.resize(0);
	}

	// Makes sure the duals are able to handle this range of dual indices for writing
	void PrepareDuals(int newDualIndexMin, int newDualIndexMax)
	{
		// if the range is empty, empty the duals
		if (newDualIndexMax < newDualIndexMin)
		{
			m_dualIndexMin = INT_MAX;
			m_dualIndexMax = INT_MIN;
			m_dual.resize(0);
		}
		// If the duals are empty, allocate
		else if (DualsEmpty())
		{
			m_dualIndexMin = (int)newDualIndexMin;
			m_dualIndexMax = (int)newDualIndexMax;
			m_dual.resize(m_dualIndexMax - m_dualIndexMin + 1, 0.0f);
		}
		// else if the duals don't already encompass this range, make sure they do, via a resize and a copy
		else if (newDualIndexMin < m_dualIndexMin || newDualIndexMax > m_dualIndexMax)
		{
			// Make a union of the new range and the old
			newDualIndexMin = std::min(newDualIndexMin, m_dualIndexMin);
			newDualIndexMax = std::max(newDualIndexMax, m_dualIndexMax);

			// allocate larger storage for the new dual values
			std::vector<float> newDual(newDualIndexMax - newDualIndexMin + 1, 0.0f);

			// copy the old values into the new area
			int oldValuesCount = m_dualIndexMax - m_dualIndexMin + 1;
			memcpy(&newDual[m_dualIndexMin - newDualIndexMin], &m_dual[0], sizeof(float) * oldValuesCount);

			// get rid of the old storage
			std::swap(m_dual, newDual);

			// update min and max
			m_dualIndexMin = (int)newDualIndexMin;
			m_dualIndexMax = (int)newDualIndexMax;
		}

		// otherwise, everything is ok as is!
	}

	// Shrinks the duals array to this size. Assumes you already made sure it's only throwing away zeros.
	void ShrinkDuals(int newDualIndexMin, int newDualIndexMax)
	{
		// The range passed in should be a subset of the old range. Enforce that
		newDualIndexMin = std::max(newDualIndexMin, m_dualIndexMin);
		newDualIndexMax = std::min(newDualIndexMax, m_dualIndexMax);

		// Nothing to do
		if (newDualIndexMin == m_dualIndexMin && newDualIndexMax == m_dualIndexMax)
			return;

		// empty
		if (newDualIndexMax < newDualIndexMin)
		{
			m_dualIndexMin = INT_MAX;
			m_dualIndexMax = INT_MIN;
			m_dual.resize(0);
			return;
		}

		// allocate storage for the new dual values
		std::vector<float> newDual(newDualIndexMax - newDualIndexMin + 1, 0.0f);

		// copy the old values into the new area
		memcpy(&newDual[0], &m_dual[newDualIndexMin - m_dualIndexMin], sizeof(float) * newDual.size());

		// get rid of the old storage
		std::swap(m_dual, newDual);

		// update min and max
		m_dualIndexMin = (int)newDualIndexMin;
		m_dualIndexMax = (int)newDualIndexMax;
	}

	void SetDualValue(int index, float f)
	{
		// make sure we can write this index
		PrepareDuals(index, index);

		// write the index
		m_dual[index - m_dualIndexMin] = f;
	}

	float GetDualValue(int index) const
	{
		// Zero where we don't have any data
		if (DualsEmpty() || index < m_dualIndexMin || index > m_dualIndexMax)
			return 0.0f;

		return m_dual[index - m_dualIndexMin];
	}

	bool DualsEmpty() const
	{
		return m_dualIndexMax < 0;
	}

	template <typename LAMBDA>
	static inline void ForEachDual(const DualNumber& A, const DualNumber& B, DualNumber& C, const LAMBDA& lambda)
	{
		// Run the foreach on the union of the duals of A and B
		int minIndex = std::min(A.m_dualIndexMin, B.m_dualIndexMin);
		int maxIndex = std::max(A.m_dualIndexMax, B.m_dualIndexMax);
		C.PrepareDuals(minIndex, maxIndex);

		#if SHRINK_DUALS()
			int minNonZero = INT_MAX;
			int maxNonZero = INT_MIN;
		#endif

		for (int i = minIndex; i <= maxIndex; ++i)
		{
			float& resultDual = C.m_dual[i - minIndex];
			lambda(A.m_real, A.GetDualValue(i), B.m_real, B.GetDualValue(i), resultDual);

			#if SHRINK_DUALS()
			if (std::abs(resultDual) > SHRINK_DUALS_ZERO_THRESHOLD())
			{
				minNonZero = std::min(minNonZero, i);
				maxNonZero = std::max(maxNonZero, i);
			}
			#endif
		}

		#if SHRINK_DUALS()
		C.ShrinkDuals(minNonZero, maxNonZero);
		#endif
	}

	template <typename LAMBDA>
	static inline void ForEachDual(const DualNumber& A, DualNumber& B, const LAMBDA& lambda)
	{
		int minIndex = A.m_dualIndexMin;
		int maxIndex = A.m_dualIndexMax;
		B.PrepareDuals(minIndex, maxIndex);

		#if SHRINK_DUALS()
			int minNonZero = INT_MAX;
			int maxNonZero = INT_MIN;
		#endif

		for (int i = minIndex; i <= maxIndex; ++i)
		{
			float& resultDual = B.m_dual[i - minIndex];
			lambda(A.m_real, A.GetDualValue(i), resultDual);

			#if SHRINK_DUALS()
			if (std::abs(resultDual) > SHRINK_DUALS_ZERO_THRESHOLD())
			{
				minNonZero = std::min(minNonZero, i);
				maxNonZero = std::max(maxNonZero, i);
			}
			#endif
		}

		#if SHRINK_DUALS()
		B.ShrinkDuals(minNonZero, maxNonZero);
		#endif
	}

	float m_real = 0.0f;
	std::vector<float> m_dual;  // A vector to not bust the stack! Also allows it to be sparse.
	int m_dualIndexMin = INT_MAX;
	int m_dualIndexMax = INT_MIN;

	//==================================================
	// Unary ops
	//==================================================

	inline DualNumber operator - () const
	{
		DualNumber ret = *this;
		ret.m_real = -m_real;
		for (float& f : ret.m_dual)
			f = -f;
		return ret;
	}

	//==================================================
	// A (op) B
	//==================================================

	inline DualNumber operator - (const DualNumber& d) const
	{
		DualNumber ret;
		ret.m_real = m_real - d.m_real;
		ForEachDual(*this, d, ret,
			[](float AReal, float ADual, float BReal, float BDual, float& retDual)
			{
				retDual = ADual - BDual;
			}
		);
		return ret;
	}

	inline DualNumber operator + (const DualNumber& d) const
	{
		DualNumber ret;
		ret.m_real = m_real + d.m_real;
		ForEachDual(*this, d, ret,
			[](float AReal, float ADual, float BReal, float BDual, float& retDual)
			{
				retDual = ADual + BDual;
			}
		);
		return ret;
	}

	inline DualNumber operator * (const DualNumber& d) const
	{
		DualNumber ret;
		ret.m_real = m_real * d.m_real;
		ForEachDual(*this, d, ret,
			[](float AReal, float ADual, float BReal, float BDual, float& retDual)
			{
				retDual = (AReal * BDual) + (ADual * BReal);
			}
		);
		return ret;
	}

	inline DualNumber operator / (const DualNumber& d) const
	{
		DualNumber ret;
		ret.m_real = m_real / d.m_real;
		ForEachDual(*this, d, ret,
			[](float AReal, float ADual, float BReal, float BDual, float& retDual)
			{
				retDual = (ADual * BReal - AReal * BDual) / (BReal * BReal);
			}
		);
		return ret;
	}
	
	//==================================================
	// A (op =) B
	//==================================================

	inline DualNumber& operator += (const DualNumber& d)
	{
		(*this) = (*this) + d;
		return *this;
	}

	inline DualNumber& operator -= (const DualNumber& d)
	{
		(*this) = (*this) - d;
		return *this;
	}

	inline DualNumber& operator *= (const DualNumber& d)
	{
		(*this) = (*this) * d;
		return *this;
	}

	inline DualNumber& operator /= (const DualNumber& d)
	{
		(*this) = (*this) / d;
		return *this;
	}
};

//==================================================
// float (op) DualNumber
// DualNumber (op) float
//==================================================

inline DualNumber operator + (float f, const DualNumber& d)
{
	DualNumber ret = d;
	ret.m_real = f + ret.m_real;
	return ret;
}

inline DualNumber operator - (float f, const DualNumber& d)
{
	DualNumber ret = d;
	ret.m_real = f - ret.m_real;
	return ret;
}

inline DualNumber operator * (float f, const DualNumber& d)
{
	DualNumber ret = d;
	ret.m_real = f * ret.m_real;
	DualNumber::ForEachDual(d, ret,
		[f](float AReal, float ADual, float& retDual)
		{
			retDual = ADual * f;
		}
	);
	return ret;
}

inline DualNumber operator / (float f, const DualNumber& d)
{
	// The DualNumber struct's division operation, but f has a dual part of zero.
	DualNumber ret;
	ret.m_real = f / d.m_real;
	DualNumber::ForEachDual(d, ret,
		[f](float AReal, float ADual, float& retDual)
		{
			retDual = (-f * ADual) / (AReal * AReal);
		}
	);
	return ret;
}

inline DualNumber operator + (const DualNumber& d, float f)
{
	DualNumber ret = d;
	ret.m_real = ret.m_real + f;
	return ret;
}

inline DualNumber operator - (const DualNumber& d, float f)
{
	DualNumber ret = d;
	ret.m_real = ret.m_real - f;
	return ret;
}

inline DualNumber operator * (const DualNumber& d, float f)
{
	DualNumber ret = d;
	ret.m_real = ret.m_real * f;
	DualNumber::ForEachDual(d, ret,
		[f](float AReal, float ADual, float& retDual)
		{
			retDual = ADual * f;
		}
	);
	return ret;
}

inline DualNumber operator / (const DualNumber& d, float f)
{
	// The DualNumber struct's division operation, but f has a dual part of zero.
	DualNumber ret;
	ret.m_real = d.m_real / f;
	DualNumber::ForEachDual(d, ret,
		[f](float AReal, float ADual, float& retDual)
		{
			retDual = ADual / f;
		}
	);
	return ret;
}

//==================================================
// std:: implementations
//==================================================

namespace std
{
	inline DualNumber exp(const DualNumber& d)
	{
		DualNumber ret;
		ret.m_real = exp(d.m_real);
		DualNumber::ForEachDual(d, ret,
			[](float AReal, float ADual, float& retDual)
			{
				retDual = ADual * std::exp(AReal);
			}
		);
		return ret;
	}
};

// More dual number operations available at: https://blog.demofox.org/2017/03/13/neural-network-gradients-backpropagation-dual-numbers-finite-differences/
