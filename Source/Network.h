#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <fstream>

#include "Random.h"
#include "Node.h"

#define WRITE_4B(i, os) os.write(reinterpret_cast<const char*>(&i), 4);
#define READ_4B(i, is) is.read(reinterpret_cast<char*>(&i), 4);

struct Network {

	Network(int inputSize, int outputSize, float* seed);

	Network(Network* n) {};
	~Network() {};

	Network(std::ifstream& is);
	
	void save(std::ofstream& os);

	// Since getOutput returns a float*, application must either use it before any other call to step(),
	// destroyPhenotype(), preTrialReset(), ... or Network destruction, either deep copy the
	// pointee immediatly when getOutput() returns. If unsure, deep copy.
	float* getOutput();

	void step(const std::vector<float>& obs);
	
	// arguments are the arrays of input sizes, output sizes, number of children and total number of nodes
	// layer by layer. nL is the number of layers (and th size of these arrays)
	void createPhenotype(int* inS, int* outS, int* nC, int* nN, int nL);
	void destroyPhenotype();

	// Sets to 0 the dynamic elements of the phenotype. 
	void preTrialReset();

	// Only used when CONTINUOUS LEARNING is not defined, in which case it updates wL with avgH.
	void postTrialUpdate();

	int inputSize, outputSize;
	
	// useful for debugging, necessary for saturation penalization
	int nInferencesOverLifetime;

	std::unique_ptr<Node> topNode;

	// Size gaussianInputVectorSize * nSubNodes.
	// nSubNodes is the phenotypic size, gaussianInputVectorSize is 
	// handled by the generator. 
	std::unique_ptr<float[]> seed;

	// Arrays for plasticity based updates. Contain all presynaptic and postSynaptic activities.
	// Must be : - reset to all 0s at the start of each trial;
	//			 - created alongside topNode_P creation; 
	//           - freed alongside topNode_P deletion.
	// Layout detailed in the Phenotype structs.
	std::unique_ptr<float[]> inputArray, destinationArray;

#ifdef STDP
	// same size and layout that of destinationArray.
	std::unique_ptr<float[]> accumulatedPreSynActs;
#endif

	// size of inputArray
	int postSynActArraySize;

	// size of destinationArray
	int preSynActsArraySize;


#ifdef SATURATION_PENALIZING
	// Sum over all the phenotype's activations, over the lifetime, of powf(activations[i], 2*n), n=typically 10.
	float saturationPenalization;

	// Follows the same usage pattern as the 4 arrays for plasticity updates. Size averageActivationArraySize.
	// Used to store, for each activation function of the phenotype, its average output over lifetime, for use in
	// getSaturationPenalization(). So set to 0 at phenotype creation, and never touched again.
	std::unique_ptr<float[]> averageActivation;

	// size of the averageActivation array.
	int averageActivationArraySize;

	float getSaturationPenalization();
#endif
};