#pragma once

#ifdef _DEBUG
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/control87-controlfp-control87-2?view=msvc-170
// These are incompatible with RocketSim that has many float errors, and should be commented when rocketsim.h and 
// .cpp are included in the project (so exclude them temporarily to use this feature).
#define _CRT_SECURE_NO_WARNINGS
#include <float.h>
unsigned int fp_control_state = _controlfp(_EM_UNDERFLOW | _EM_INEXACT, _MCW_EM);

#endif

#include "System.h"

#ifdef ROCKET_SIM_T
#include "RocketSim.h"
#endif


#include <eigen-3.4.0/Eigen/Core>

using namespace std;


int main()
{
#ifdef _OPENMP
    //// https://docs.huihoo.com/eigen/3/TopicMultiThreading.html
    //Eigen::initParallel();
    //int nThreads = Eigen::nbThreads();
    //omp_set_num_threads(nThreads);
    //Eigen::setNbThreads(nThreads);
#endif



    LOG_("Seed : " << seed);


    if (torch::cuda::is_available()) {
        std::cout << "CUDA available for libtorch !" << std::endl;
    }



#ifdef ROCKET_SIM_T
    // Path to where you dumped rocket league collision meshes.
    RocketSim::Init((std::filesystem::path)"C:/Users/alpha/Bureau/RLRL/collisionDumper/x64/Release/collision_meshes");
#endif



    Trial* trial;

#ifdef CARTPOLE_T
    trial = new CartPoleTrial(true); // bool : continuous control.
#elif defined XOR_T
    trial = new XorTrial(4, 5);  // int : vSize, int : delay
#elif defined TEACHING_T
    trial = new TeachingTrial(5, 5);
#elif defined TMAZE_T
    trial = new TMazeTrial(false);
#elif defined N_LINKS_PENDULUM_T
    trial = new NLinksPendulumTrial(false, 2);
#elif defined MEMORY_T
    trial = new MemoryTrial(2, 5, 5, true); // int nMotifs, int motifSize, int responseSize, bool binary = true
#elif defined ROCKET_SIM_T
    trial = new RocketSimTrial();
#endif


    int trialObservationsSize = trial->netInSize;
    int trialActionsSize = trial->netOutSize;


#define TRIVIAL_ARCHITECTURE
#ifdef TRIVIAL_ARCHITECTURE
    const int treeDepth = 1;
    int inSizes[treeDepth] = { trialObservationsSize };
    int outSizes[treeDepth] = { trialActionsSize };
    int nChildrenPerLayer[treeDepth] = { 0 }; // Must end with 0
    int nEvolvedModulesPerLayer[treeDepth] = { 64 };
#else
    // A structurally non trivial example
    const int treeDepth = 2;
    int inSizes[treeDepth] = { trialObservationsSize, 4 };
    int outSizes[treeDepth] = { trialActionsSize, 3 };
    int nChildrenPerLayer[treeDepth] = { 2, 0 }; // Must end with 0
    int nEvolvedModulesPerLayer[treeDepth] = { 32, 64 };
#endif

#ifdef ACTION_L_OBS_O
    {
        int temp = inSizes[0];
        inSizes[0] = outSizes[0];
        outSizes[0] = temp;
    }
#endif





    SystemParameters sParams;


    sParams.learningRate = .1f;
    sParams.nBatches = 2;
    sParams.nPerturbations = 5;
    sParams.nEvaluationTrials = 4;
    sParams.nSupervisedTrials = 4;
    sParams.nTeachers = 4;
    sParams.seedSize = 5;

    sParams.treeDepth = treeDepth;
    sParams.inSizePerL = inSizes;
    sParams.outSizePerL = outSizes;
    sParams.nChildrenPerL = nChildrenPerLayer;


    int nSteps = 10000;


    System system(sParams, trial);

    system.evolve(nSteps);

    return 0;
}
