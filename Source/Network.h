#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <fstream>

#include "DeMoCEvoCore.h"
#include "Node.h" 


struct Network {

	static int activationArraySize;
	static int* inS;
	static int* outS;
	static int* nC;
	static int nLayers;

	Network();

	Network(const Network& pcn);

	~Network() {};

	Network(std::ifstream& is);
	
	void save(std::ofstream& os);

	// Since getOutput returns a float*, application must either use it before any other call to step(),
	// destroyPhenotype(), preTrialReset(), ... or Network destruction, either deep copy the
	// pointee immediatly when getOutput() returns. If unsure, deep copy.
	float* getOutput();

	void step(float* input, bool supervised, float* target = nullptr);
	
	void setInitialActivations(float* initialActivations) { std::copy(initialActivations, initialActivations + activationArraySize, activations.get()); };


	void createPhenotype(GeneratorNode* rootGenerator, int perturbationID, bool negative);

	void destroyPhenotype();

	// Sets to 0 the dynamic elements of the phenotype. 
	void preTrialReset();


	std::unique_ptr<Node> rootNode;


	// The following arrays hold per-neuron quantities for the whole network. 
	// Each node of the tree has its own pointers inside each array to its 
	// dedicated, "personal", storage. Done this way to minimize cache misses
	// and memory allocations.
	std::unique_ptr<float[]> activations;
	std::unique_ptr<float[]> accumulators;

};