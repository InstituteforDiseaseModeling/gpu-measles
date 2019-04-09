/*
** copyright banner
*/

#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <climits>
#include <numeric>

#include "cuda.h"
#include <curand_kernel.h>
#include "cub/cub/cub.cuh"

#include "block-size.h"
#include "configuration.h"
#include "demographics.h"
#include "distributions.h"
#include "measles-kernel.h"
#include "spatial-tx.h"
#include "utilities.h"
#include "vax-kernel.h"

bool debug = false;

// crude birthrate: 40/1000/year = 0.04/individual/year  = 0.00010958904109589041/individual/day
// crude mortality: 15/1000/year = 0.015/individual/year
//                               = 0.985/individual/year survival probability
//                               = 0.99995859363039054212/individual/day survival probability
//                               = 0.00004140636960945788/individual/day mortality probability

#define BIRTHRATE   (0.0001096f)
#define MORTALITY   (0.0000414f)

int main(int argc, const char* argv[])
{
    SimConfig configuration;
    configuration.Initialize( argc, argv );

    Census census;
    census.Initialize( configuration );
    PopulationImpl  population( 1 << 28, configuration.prng_seed ); // 256 million individuals (maximum capacity)
    population.Initialize( census );

    std::cerr << "Running " << population.size << " agents for " << configuration.sim_duration << " time steps. " << std::endl;

    Transmission* spatialTransmission = TransmissionFromFile( configuration.network_file_name, census.num_locales );

    StatisticsImpl statistics( configuration.sim_duration + 1, census.num_locales, census.population, census.prevalence );

    auto start = clock();
    for ( size_t step = 0; step < configuration.sim_duration; ++step )
    {
        Statistics current = statistics + step;
        Statistics next    = current + 1;
        population.Update( current, next );

        // Contagion went into next but population does not get updated until VitalDynamics below.
        spatialTransmission->DoTransmission( next.contagion, next.contagion, current.population );
        population.VitalDynamics( BIRTHRATE, MORTALITY, 1.0f, current, next );
    }
    auto end = clock();
    util::PrintElapsedTime(start, end, "Simulation");

    statistics.WriteToFile( "output.csv" );
}
