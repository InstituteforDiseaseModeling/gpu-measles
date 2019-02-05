#include <cuda_runtime.h>
#include <curand.h>

#include <curand_mtgp32_host.h>

#include <iostream>

#include "block-size.h"
#include "../gpu/check-return.h"

#include "prng-kernel.h"

void AllocateVariateMemory(float** ppVariates, uint32_t count);
void AllocatePrngState(curandState** ppPrngState, uint32_t count);

void DisplayResults(uint32_t* bins, uint32_t NUM_BINS, float* pVariates, uint32_t count, float scale, float offset, const char* header);
void Histogram(float* pVariates, float scale, float offset, uint32_t count, uint32_t* bins, uint32_t num_bins);
void DumpBins(uint32_t* bins, uint32_t num_bins);
void PlotBins(uint32_t* bins, uint32_t num_bins);

void AllocateMTGP32State(curandStateMtgp32** ppMTGP32States, uint32_t num_states);
void AllocateKernelParams(mtgp32_kernel_params** ppKernelParams);
void MakeMtgp32KernelState(curandStateMtgp32* pMTGP32States, mtgp32_params_fast_t mtgp32dc_params_fast_11213[], mtgp32_kernel_params* pKernelParams, uint32_t num_states, uint64_t seed, uint64_t delta);

void DisplayElapsedTime(clock_t start, clock_t end, const char* msg = nullptr, std::ostream& stream = std::cerr);

int main(int argc, const char* argv[])
{
    float* pVariates = nullptr;
    uint32_t    count = 32*1024*1024;

    if ( argc > 1 )
    {
        count = uint32_t(atol(argv[1]));
        std::cout << "count is " << count << std::endl;
    }

    std::cout << "FYI, sizeof(curandState) is ... " << sizeof(curandState) << " bytes." << std::endl << std::endl;

    AllocateVariateMemory(&pVariates, count);

    // ***** "default" PRNG *****

    curandState* pPrngState = nullptr;
    AllocatePrngState(&pPrngState, count);
    // BLOCK_SIZE is number of threads in a single block
    uint32_t num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto start = clock();
    initialize_prng<<<num_blocks, BLOCK_SIZE>>>(pPrngState, count);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
    auto end = clock();
    DisplayElapsedTime(start, end, "initialize_prng (xorwow)");

    #define NUM_BINS    (32)
    uint32_t bins[NUM_BINS];

    // uniform variates
    start = clock();
    uniform_default<<<num_blocks, BLOCK_SIZE>>>(pPrngState, pVariates, count);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
    end = clock();
    DisplayElapsedTime(start, end, "uniform_xorwow    ");
    DisplayResults(bins, NUM_BINS, pVariates, count, NUM_BINS, 0.0f, "===== Uniform distribution (0-32) with default PRNG: =====");

    // Gaussian variates
    start = clock();
    gaussian_default<<<num_blocks, BLOCK_SIZE>>>(pPrngState, pVariates, count);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
    end = clock();
    DisplayElapsedTime(start, end, "gaussian_xorwow   ");
    DisplayResults(bins, NUM_BINS, pVariates, count, 4.0f, 16.0f, "===== Gaussian (normal) distribution (mean = 16, ? = 4) with default PRNG: =====");

    // exponential variates
    start = clock();
    exponential_default<<<num_blocks, BLOCK_SIZE>>>(pPrngState, 8.0f, pVariates, count);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
    end = clock();
    DisplayElapsedTime(start, end, "exponential_xorwow");
    DisplayResults(bins, NUM_BINS, pVariates, count, 1.0f, 0.0f, "===== Exponential distribution (mean = 8) with default PRNG: =====");

    // Poisson variates
    start = clock();
    poisson_default<<<num_blocks, BLOCK_SIZE>>>(pPrngState, 8.0f, pVariates, count);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
    end = clock();
    DisplayElapsedTime(start, end, "poisson_xorwow    ");
    DisplayResults(bins, NUM_BINS, pVariates, count, 1.0f, 0.0f, "===== Poisson distribution (λ = 8) with default PRNG: =====");

    // ***** MTGP32 PRNG *****

    curandStateMtgp32* pMTGP32States;
    uint32_t num_states = (count + 255) / 256;
    AllocateMTGP32State(&pMTGP32States, num_states);
    mtgp32_kernel_params* pKernelParams;
    AllocateKernelParams(&pKernelParams);
    CheckCurand(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, pKernelParams));
    MakeMtgp32KernelState(pMTGP32States, mtgp32dc_params_fast_11213, pKernelParams, num_states, 20190102, 9999959999);

    // uniform variates
    start = clock();
    uniform_mtgp32<<<num_blocks, BLOCK_SIZE>>>(pMTGP32States, pVariates, count);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
    end = clock();

    for (uint32_t i = 0; i < 8; ++i)
        std::cout << pVariates[i] << std::endl;
    std::cout << std::endl;

    for (uint32_t i = 0; i < 8; ++i)
    {
        // std::cout << pVariates[i*256] << std::endl;
        for (uint32_t j = 0; j < 4; ++j)
        {
            std::cout << pVariates[i*200+j] << std::endl;
        }
    }
    std::cout << std::endl;

    DisplayElapsedTime(start, end, "uniform_mtgp32    ");

    DisplayResults(bins, NUM_BINS, pVariates, count, NUM_BINS, 0.0f, "===== Uniform distribution (0-32) with MTGP32 PRNG: =====");

    // Gaussian variates
    start = clock();
    gaussian_mtgp32<<<num_blocks, BLOCK_SIZE>>>(pMTGP32States, pVariates, count);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
    end = clock();
    DisplayElapsedTime(start, end, "gaussian_mtgp32   ");
    DisplayResults(bins, NUM_BINS, pVariates, count, 4.0f, 16.0f, "===== Gaussian (normal) distribution (mean = 16, ? = 4) with MTGP32 PRNG: =====");

    // exponential variates
    start = clock();
    exponential_mtgp32<<<num_blocks, BLOCK_SIZE>>>(pMTGP32States, 8.0f, pVariates, count);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
    end = clock();
    DisplayElapsedTime(start, end, "exponential_mtgp32");
    DisplayResults(bins, NUM_BINS, pVariates, count, 1.0f, 0.0f, "===== Exponential distribution (mean = 8) with MTGP32 PRNG: =====");

    // Poisson variates
    start = clock();
    poisson_mtgp32<<<num_blocks, BLOCK_SIZE>>>(pMTGP32States, 8.0f, pVariates, count);
    CheckReturn(cudaDeviceSynchronize());
    CheckReturn(cudaGetLastError());
    end = clock();
    DisplayElapsedTime(start, end, "poisson_mtgp32    ");
    DisplayResults(bins, NUM_BINS, pVariates, count, 1.0f, 0.0f, "===== Poisson distribution (λ = 8) with MTGP32 PRNG: =====");

    CheckReturn(cudaFree(pKernelParams));
    CheckReturn(cudaFree(pMTGP32States));
    CheckReturn(cudaFree(pPrngState));
    CheckReturn(cudaFree(pVariates));
}

void AllocateVariateMemory(float** ppVariates, uint32_t count)
{
    CheckReturn(cudaMallocManaged(ppVariates, count * sizeof(float)));
}

void AllocatePrngState(curandState** ppPrngState, uint32_t count)
{
    CheckReturn(cudaMallocManaged(ppPrngState, count * sizeof(curandState)));
    CheckReturn(cudaMemset(*ppPrngState, 0, count * sizeof(curandState)));
}

void Histogram(float* pVariates, float scale, float offset, uint32_t count, uint32_t* bins, uint32_t num_bins)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t bin = uint32_t((pVariates[i] * scale) + offset);
        if (bin < num_bins) // Don't need to check for >= 0 because bin is unsigned.
        {
            ++bins[bin];
        }
    }
}

void DumpBins(uint32_t* bins, uint32_t num_bins)
{
    for (uint32_t bin = 0; bin < num_bins; ++bin)
    {
        std::cout << bin << ',' << bins[bin] << std::endl;
    }
    std::cout << std::endl;
}

void PlotBins(uint32_t* bins, uint32_t num_bins)
{
    bool more = true;

    while (more)
    {
        more = false;
        for (uint32_t bin = 0; bin < NUM_BINS; ++bin)
        {
            if (bins[bin] > 0)
            {
                std::cout << '*';
                --bins[bin];
                more = true;
            }
            else
            {
                std::cout << ' ';
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void DisplayResults(uint32_t* bins, uint32_t num_bins, float* pVariates, uint32_t count, float scale, float offset, const char* header)
{
    if ( count <= 1024 )
    {
        memset((void*)bins, 0, num_bins * sizeof(uint32_t));
        Histogram(pVariates, scale, offset, count, bins, num_bins);
        std::cout << header << std::endl;
        DumpBins(bins, num_bins);
        PlotBins(bins, num_bins);
    }
}

void AllocateMTGP32State(curandStateMtgp32** ppMTGP32States, uint32_t num_states)
{
    std::cout << "FYI, sizeof(curandStateMtgp32) = " << sizeof(curandStateMtgp32) << " bytes." << std::endl;
    CheckReturn(cudaMallocManaged(ppMTGP32States, num_states * sizeof(curandStateMtgp32)));
}

void AllocateKernelParams(mtgp32_kernel_params** ppKernelParams)
{
    CheckReturn(cudaMalloc(ppKernelParams, sizeof(mtgp32_kernel_params)));
}

void MakeMtgp32KernelState(curandStateMtgp32* pMTGP32States, mtgp32_params_fast_t mtgp32dc_params_fast_11213[], mtgp32_kernel_params* pKernelParams, uint32_t num_states, uint64_t seed, uint64_t delta)
{
    for (uint32_t i = 0; i < num_states; i += 200)
    {
        uint32_t count = min(num_states - i, 200);
        CheckCurand(curandMakeMTGP32KernelState(pMTGP32States + i, mtgp32dc_params_fast_11213, pKernelParams, count, seed));
        seed += delta;
    }
}

void DisplayElapsedTime(clock_t start, clock_t end, const char* msg, std::ostream& stream)
{
    if ( msg )
    {
        stream << msg << " : ";
    }
    stream << (1000 * uint64_t(end - start) / double(CLOCKS_PER_SEC)) << " milliseconds" << std::endl;
}
