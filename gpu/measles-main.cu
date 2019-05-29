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
#include "cmdline.h"
#include "config-json.h"
#include "constants.h"
#include "demographics.h"
#include "distributions.h"
#include "measles-kernel.h"
#include "spatial-tx.h"
#include "utilities.h"
#include "vax-kernel.h"
#include "importation.h"

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
    CmdLine cmdLine( argc, argv );
    ConfigJson configJson( cmdLine.config_filename );
    const ConfigParameters& parameters = configJson.parameters;

    Constants constants;
    constants.incubation_duration_minimum = parameters.incubation_Period_Min;
    constants.incubation_duration_maximum = parameters.incubation_Period_Max;
    constants.infection_duration_minimum  = parameters.infectious_Period_Min;
    constants.infection_duration_maximum  = parameters.infectious_Period_Max;
    constants.infectivity                 = parameters.base_Infectivity;
    constants.UpdateDevice();

    Census census;
    census.Initialize( parameters );
    PopulationImpl  population( 1 << 28, parameters.run_Number ); // 256 million individuals (maximum capacity)
    float r0 = ((constants.infection_duration_minimum + constants.infection_duration_maximum) / 2.0f) * constants.infectivity;
    population.Initialize( census, parameters.enable_Immunity && parameters.enable_Initial_Susceptibility_Distribution, r0, MORTALITY );

    std::cerr << "Running " << population.size << " agents for " << parameters.simulation_Duration << " time steps. " << std::endl;

    Transmission* spatialTransmission = TransmissionFromFile( parameters.regional_Migration_Filename, census.num_locales, parameters.x_Regional_Migration );

    StatisticsImpl statistics( parameters.simulation_Duration + 1, census.num_locales, census.population, census.prevalence );
    TransmissionLogImpl transmissions( 1 << 28 );

    Importation* importation = new ContagionIncrease( 30, std::vector<uint16_t>({ 313, 655, 635, 24, 182, 445 }), 20.0f );

    auto start = clock();
    for ( size_t step = 0; step < parameters.simulation_Duration; ++step )
    {
        Statistics current = statistics + step;
        Statistics next    = current + 1;
        population.Update( uint32_t(step), current, next, transmissions );

        // Contagion went into next but population does not get updated until VitalDynamics below.
        spatialTransmission->DoTransmission( next.contagion, next.contagion, current.population );
        population.VitalDynamics( BIRTHRATE, MORTALITY, 1.0f, current, next );
        importation->Apply( step, population, statistics );
    }
    auto end = clock();
    util::PrintElapsedTime(start, end, "Simulation");

    statistics.WriteToFile( "output.csv" );
    transmissions.WriteToFile( "transmission.bin" );
}
