#pragma once


////////////////////////////////////
///// USER COMPILATION CHOICES /////
////////////////////////////////////


// Comment/uncomment or change the value of various preprocessor directives
// to compile different versions of the code. Or use the -D flag.

#pragma once

////////////////////////////////////
///// USER COMPILATION CHOICES /////
////////////////////////////////////


// Comment/uncomment or change the value of various preprocessor directives
// to compile different versions of the code. Or use the -D flag.


// minor options still in the files : 
// -trial uses same seed at reset (in system.cpp, thread loop).
// -value of initial activations at agent creation (normal * .3)
// -formula for ranking scores. (functions.cpp, rankArray())


// Use custom CUDA kernels instead of libtorch's (GPU or CPU) BLAS. 
//#define CUSTOM_KERNELS


// Choose the trial the algorithm will (try to) solve. One and only one must be defined.
#define CARTPOLE_T
//#define XOR_T
//#define TEACHING_T
//#define TMAZE_T
//#define N_LINKS_PENDULUM_T
//#define MEMORY_T
//#define ROCKET_SIM_T 

#ifdef CARTPOLE_T
#define COPY CartPoleTrial
#elif defined XOR_T
#define COPY XorTrial
#elif defined TEACHING_T
#define COPY TeachingTrial
#elif defined TMAZE_T
#define COPY TMazeTrial
#elif defined N_LINKS_PENDULUM_T
#define COPY  NLinksPendulumTrial
#elif defined MEMORY_T
#define COPY  MemoryTrial
#elif defined ROCKET_SIM_T
#define COPY  RocketSimTrial
#endif


// when defined, there is only one internal connexion per module, it is no longer split into toOutput and the vector of toChildren.
// It has a performance boost and makes code shorter and more maintainable, but has 2 downsides.
// - Requires out,Cin be adjacent. But the child module requires its in,Cout be adjacent. So extra activation copy.
// - No flexibility for the order of propagation of information through the network, it necessarily simultaneous.
#define ONE_MATRIX


#define SAME_SEED

// Instead of a fixed, global variance equal to 1 across the activations, modules and agents, 
// each sigma is now evolved and per activation. (But also dynamically updated, as thetas and biases.)
// This can be interpred as using a per activation variance not fixed to 1, but without correlation 
// between activations on the same layer. It is used to multiply epsilons before any further computations.
// The vector contains the inverses to speed up inference.
// As in Friston (2005), invSigma takes values in [0,1], i.e. sigma >= 1. The values obtained from the generators
// are changed in ConnexionGenerator::createPhenotypeArrays (generatorNode.cpp).
// TODO insert modulation ----------------------------------------------------------------------------- TODO
#define ACTIVATION_VARIANCE


// Still looking for a satisfactory solution. 
#define MODULATED

// Places the observation at layer 0, and the action at layer L. (normally the other way around)
// A network that is not reversed, not modulated, and whose root module does not have children, is effectively a fixed-weight one.
//#define ACTION_L_OBS_O



//******************* END OF PARAMETERS CHOICES ***************//

// what follows must not be modified, it computes the required memory space for the algorithm
// (by computing how many parameters each module requires) and other logic


#ifdef ACTIVATION_VARIANCE
#define SIGMA_VEC 1
#else
#define SIGMA_VEC 0
#endif

#ifdef MODULATED
#define MOD_MAT 1
#else
#define MOD_MAT 0
#endif

#define N_MATRICES (1 + MOD_MAT)


#define N_VECTORS (1 + SIGMA_VEC)