#pragma once

#ifdef _DEBUG
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/control87-controlfp-control87-2?view=msvc-170
// These are incompatible with RocketSim that has many float errors, and should be commented when rocketsim.h and 
// .cpp are included in the project (so exclude them temporarily to use this feature).
#define _CRT_SECURE_NO_WARNINGS
#include <float.h>
unsigned int fp_control_state = _controlfp(_EM_UNDERFLOW | _EM_INEXACT, _MCW_EM);

#endif

#include <iostream>
#include "Generator.h"

#ifdef ROCKET_SIM_T
#include "RocketSim.h"
#endif


#define LOGV(v) for (const auto e : v) {cout << std::setprecision(2)<< e << " ";}; cout << "\n"

using namespace std;
using namespace torch::indexing;

int main()
{

    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
    }
    else {
        std::cout << "Training on CPU." << std::endl;
    }
    

    cout << "Seed : " << seed << endl;

#ifdef ROCKET_SIM_T
    // Path to where you dumped rocket league collision meshes.
    RocketSim::Init((std::filesystem::path)"C:/Users/alpha/Bureau/RLRL/collisionDumper/x64/Release/collision_meshes");
#endif

    int nSteps = 10000;


    CartPoleTrial t(false);
    GeneratorParameters params;
   

    Generator generator(t.netInSize, t.netOutSize);
    generator.setParameters(params);



    // Main loop
    for (int i = 0; i < nSteps; i++) {



#ifdef ROCKET_SIM_T
        if (i < 10.0f) {
            float jbt[3] = { .002f, .002f, .002f };
            for (int j = 0; j < nDifferentTrials; j++) {
                trials[j]->outerLoopUpdate(&jbt);
            }
        }
#endif

        generator.step(&t);

        if ((i + 1) % 20 == 0) {
            generator.save1Net();
        }
    }


    return 0;
}
