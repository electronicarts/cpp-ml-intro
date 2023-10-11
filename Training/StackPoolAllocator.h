///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>
#include <span>

// A stack allocator that is also strongly typed

template <typename T>
class StackPoolAllocator
{
public:
	StackPoolAllocator(size_t maxCount)
	{
		// Max size must be set in advance to not make pointers invalid while this is in use.
		m_storage.resize(maxCount);
	}

	inline void Reset()
	{
		m_nextFree = 0;
	}

	template <size_t COUNT, bool INITIALIZE>
	inline std::span<T, COUNT> Allocate()
	{
		// Allocate space
		T* ret = &m_storage[m_nextFree];
		m_nextFree += COUNT;

		// nullptr when we run out of space.
		if (m_nextFree > m_storage.size())
			return std::span<T, COUNT>{ (T*)nullptr, COUNT };

		// Make the values clean
		auto retSpan = std::span<T, COUNT>{ ret, COUNT };

		if (INITIALIZE)
		{
			T dflt = {};
			std::fill(retSpan.begin(), retSpan.end(), dflt);
		}

		// Return the data
		return retSpan;
	}

	inline std::span<T> Allocate(size_t count, bool initialize)
	{
		// Allocate space
		T* ret = &m_storage[m_nextFree];
		m_nextFree += count;

		// nullptr when we run out of space.
		if (m_nextFree > m_storage.size())
			return std::span<T>{ (T*)nullptr, 0 };

		// Make the values clean
		auto retSpan = std::span<T>{ ret, count };
		if (initialize)
		{
			T dflt = {};
			std::fill(retSpan.begin(), retSpan.end(), dflt);
		}

		// Return the data
		return retSpan;
	}

private:
	size_t m_nextFree = 0;
	std::vector<T> m_storage;
};
