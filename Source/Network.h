#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <fstream>

#include "DeMoCEvoCore.h"
#include "Module.h" 


struct Network {

	static int activationArraySize;
	static int* inS;
	static int* outS;
	static int* nC;
	static int nLayers;

	// cpu if cuda is not used/available, GPU otherwise.
	torch::Device* device;

	// Allocate memory for modules, and arrays managed by this.
	Network(GeneratorNode* rootGenerator);

	// Fills pre-allocated memory with data derived from the generated genotype.
	void generatePhenotype(GeneratorNode* rootGenerator, int perturbationID, bool negative);

	void deepCopy(const Network& pcn); 

	~Network() 
	{
		if (*device == torch::kCUDA) {
			cudaFree(activations);
			cudaFree(accumulators);
		}
		else {
			delete[] activations;
			delete[] accumulators;
		}
	};

	Network(std::ifstream& is);
	
	void save(std::ofstream& os);

	// Since getOutput returns a float*, application must either use it before any other call to step(),
	// destroyPhenotype(), preTrialReset(), ... or Network destruction, either deep copy the
	// pointee immediatly when getOutput() returns. If unsure, deep copy.
	float* getOutput();

	void step(float* input, bool supervised, float* target = nullptr);
	
	// Sets to 0 the dynamic elements of the phenotype. 
	void preTrialReset();


	std::unique_ptr<Module> rootModule;


	// The following arrays hold per-neuron quantities for the whole network. 
	// Each node of the tree has its own pointers inside each array to its 
	// dedicated, "personal", storage. Done this way to minimize cache misses
	// and memory allocations. Memory is on CPU or GPU depending on the device used.
	float* activations;
	float* accumulators;

	// are mapped over the activations and accumulators pointers.
	torch::Tensor activationsTensor;
	torch::Tensor accumulatorsTensor;

};