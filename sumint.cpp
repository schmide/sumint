#include <iostream>
#include <immintrin.h>
#include <inttypes.h>

int64_t Sum8(int8_t *buffer, uint64_t count, int8_t sanitizeValue = -128)
{
	if (!count)
		return 0;
	int64_t total = 0;
	int8_t *end = buffer + count;
	do {
		if (*buffer != sanitizeValue)
			total += *buffer;
	} while (++buffer < end);
	return total;
}

int64_t Sum32(int32_t *buffer, uint64_t count)
{
	if (!count)
		return 0;
	int64_t total = 0;
	int32_t *end = buffer + count;
	do {
		total += *buffer;
	} while (++buffer < end);
	return total;
}

#define MASK(a) ( ( 1 << (a) ) - 1 )
#define MASKCAST(a,b) ( ( ( (a)1 ) << (b) ) - 1 )
#define TOALIGNED32(a,b) ( ((unsigned long)(-(long)reinterpret_cast<long long>(a))) & MASK(b) )
#define TOALIGNED64(a,b) ( ((unsigned long long)(-reinterpret_cast<long long>(a))) & MASKCAST(unsigned long long, b) )

int64_t SumAVX8(int8_t *buffer, uint64_t count, int8_t sanitizeValue = -128)
{
	if (!count)
		return 0;
	int64_t total = 0;
	int64_t leadIn = TOALIGNED64(buffer, 4); // to align 16
	if (count - leadIn < 32)
		leadIn = count;
	if (leadIn) {
		int8_t *end8 = buffer + leadIn;
		do {
			if (*buffer != sanitizeValue)
				total += *buffer;
		} while (++buffer < end8);
		count -= leadIn;
	}
	if (!count)
		return total;
	__m256i bytem128 = _mm256_set1_epi8(sanitizeValue);
	int64_t runs = count >> 13;
	int64_t firstRun = (count >> 5) & MASK(8);
	count -= (runs << 13) + (firstRun << 5);
	__m256i *runner = reinterpret_cast<__m256i *>(buffer);
	__m256i *end = runner + (runs << 8) + firstRun;
	uint64_t subRun = firstRun ? firstRun : 1 << 8;
	do {
		__m256i *endRun = runner + subRun;
		__m256i wordSum = _mm256_set1_epi16(0);
		do {
			__m256i sanatized = _mm256_sub_epi8(*runner, _mm256_and_si256(*runner, _mm256_cmpeq_epi8(*runner, bytem128)));
			wordSum = _mm256_add_epi16(wordSum, _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(sanatized)), _mm256_cvtepi8_epi16(_mm256_extracti128_si256(sanatized, 1))));
		} while (++runner < endRun);
		__m256i dwordSum = _mm256_hadd_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(wordSum)), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(wordSum, 1)));
		__m128i vsum = _mm_add_epi32(_mm256_castsi256_si128(dwordSum), _mm256_extracti128_si256(dwordSum, 1));
		vsum = _mm_add_epi32(vsum, _mm_srli_si128(vsum, 8));
		vsum = _mm_add_epi32(vsum, _mm_srli_si128(vsum, 4));
		total += _mm_cvtsi128_si32(vsum);
		subRun = 1 << 8;
	} while (runner < end);
	int remainder = count & 31;
	if (remainder) {
		buffer = reinterpret_cast<int8_t *>(runner);
		int8_t *end8 = buffer + remainder;
		do {
			if (*buffer != -128)
				total += *buffer;
		} while (++buffer < end8);
	}
	return total;
}
int64_t SumAVX32(int32_t *buffer, uint64_t count)
{
	if (!count)
		return 0;
	int64_t total = 0;
	uint64_t leadIn = TOALIGNED64(buffer, 4) >> 2; // to align 16
	if (count - leadIn < 8)
		leadIn = count;
	if (leadIn) {
		int32_t *end32 = buffer + leadIn;
		do {
			total += *buffer;
		} while (++buffer < end32);
		count -= leadIn;
	}
	if (!count)
		return total;
	uint64_t runs = count >> 3;
	count -= (runs << 3);
	__m256i *runner = reinterpret_cast<__m256i *>(buffer);
	__m256i *end = runner + runs;
	__m256i qwordSum = _mm256_set1_epi64x(0);
	do {
		qwordSum = _mm256_add_epi64(qwordSum, _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(*runner)), _mm256_cvtepi32_epi64(_mm256_extracti128_si256(*runner, 1))));
	} while (++runner < end);
	__m128i vsum = _mm_add_epi64(_mm256_castsi256_si128(qwordSum), _mm256_extracti128_si256(qwordSum, 1));
	vsum = _mm_add_epi64(vsum, _mm_srli_si128(vsum, 8));
	total += _mm_cvtsi128_si64(vsum);
	int remainder = count & 15;
	if (remainder) {
		buffer = reinterpret_cast<int32_t *>(runner);
		int32_t *end32 = buffer + remainder;
		do {
			total += *buffer;
		} while (++buffer < end32);
	}
	return total;
}


int main()
{
	// good test case. there is one -128 per 256 so a 256 to one ratio
	// also with -128 as the sanitize value. zero is the correct output.

#define POWER 20
#define OFFSET 0
#define SUMMERSIZE8 (1<<POWER)
#define SUMMERSIZE32 (1<<(POWER-8))
	alignas(16) int8_t *sumData8 = new int8_t[SUMMERSIZE8 + OFFSET];
	alignas(16) int32_t *sumData32 = new int32_t[SUMMERSIZE32 + OFFSET];
	for (int i = 0; i < SUMMERSIZE8; i++) {
		sumData8[i + OFFSET] = i;
	}
	for (int i = 0; i < SUMMERSIZE32; i++) {
		sumData32[i + OFFSET] = i;
	}
	int64_t summer8 = SumAVX8(sumData8 + OFFSET, SUMMERSIZE8);
	int64_t summer32 = SumAVX32(sumData32 + OFFSET, SUMMERSIZE32);
	int64_t summer8x = Sum8(sumData8 + OFFSET, SUMMERSIZE8);
	int64_t summer32x = Sum32(sumData32 + OFFSET, SUMMERSIZE32);
	printf("% " PRId64 "\n", summer8);
	printf("% " PRId64 "\n", summer32);
	printf("% " PRId64 "\n", summer8x);
	printf("% " PRId64 "\n", summer32x);
	delete[] sumData8;
	delete[] sumData32;
}



///////////////////////////////////////


int64_t SumAVX8x(int8_t *buffer, uint64_t count, int8_t sanitizeValue = -128)
{
	if (!count)
		return 0;
	int64_t total = 0;
	int64_t leadIn = TOALIGNED64(buffer, 4); // to align 16
	if (count - leadIn < 32)
		leadIn = count;
	if (leadIn) {
		int8_t *end8 = buffer + leadIn;
		do {
			if (*buffer != sanitizeValue)
				total += *buffer;
		} while (++buffer < end8);
		count -= leadIn;
	}
	if (!count)
		return total;
	__m256i bytem128 = _mm256_set1_epi8(sanitizeValue);
	int64_t runs = count >> 13;
	int64_t firstRun = (count >> 5) & MASK(8);
	count -= (runs << 13) + (firstRun << 5);
	__m256i *runner = reinterpret_cast<__m256i *>(buffer);
	__m256i *end = runner + (runs << 8) + firstRun;
	uint64_t subRun = firstRun ? firstRun : 1 << 8;
	do {
		__m256i *endRun = runner + subRun;
		__m256i wordSum = _mm256_set1_epi16(0);
		do {
			__m256i sanatized = _mm256_sub_epi8(*runner, _mm256_and_si256(*runner, _mm256_cmpeq_epi8(*runner, bytem128)));
			__m128i *sanatized128 = reinterpret_cast<__m128i *>(&sanatized);
			wordSum = _mm256_add_epi16(wordSum, _mm256_add_epi16(_mm256_cvtepi8_epi16(*sanatized128), _mm256_cvtepi8_epi16(sanatized128[1])));
		} while (++runner < endRun);
		__m128i *wordSum128 = reinterpret_cast<__m128i *>(&wordSum);
		__m256i dwordSum = _mm256_hadd_epi32(_mm256_cvtepi16_epi32(*wordSum128), _mm256_cvtepi16_epi32(wordSum128[1]));
		int32_t *vsum = reinterpret_cast<int32_t *>(&dwordSum);
		int32_t *vsumEnd = vsum + 8;
		do {
			total += *vsum++;
		} while (vsum < vsumEnd);
		subRun = 1 << 8;
	} while (runner < end);
	int remainder = count & 31;
	if (remainder) {
		buffer = reinterpret_cast<int8_t *>(runner);
		int8_t *end8 = buffer + remainder;
		do {
			if (*buffer != -128)
				total += *buffer;
		} while (++buffer < end8);
	}
	return total;
}

