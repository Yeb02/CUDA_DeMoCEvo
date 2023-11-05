#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "InternalConnexion.h"
#include "GeneratorNode.h"

#include <cuda_runtime.h>


struct Module {

	int inputSize, outputSize, nChildren;   

	// cpu if cuda is not used/available, GPU otherwise.
	torch::Device* device;

	std::vector<Module> children;

#ifdef ONE_MATRIX
	InternalConnexion parameters;
#else
	std::vector<InternalConnexion> toChildren;
	InternalConnexion toOutput;
#endif

	// Concatenation of this node's input and children's output.
	torch::Tensor inCoutActivations;
	torch::Tensor inCoutAccumulators;

	// This node's input
	torch::Tensor inputActivations;
	torch::Tensor inputAccumulators;

	// This node's output.
	torch::Tensor outputActivations;
	torch::Tensor outputAccumulators;

#ifdef ONE_MATRIX
	torch::Tensor outCinActivations;
	torch::Tensor outCinAccumulators;
#endif


	// Allocate memory for connexions, and arrays managed by this.
	Module(GeneratorNode& generator);

	// Fills pre-allocated memory with data derived from the generated genotype.
	void generatePhenotype(GeneratorNode& generator, int perturbationID, bool negative);

	// Should never be called.
	Module() :
#ifdef ONE_MATRIX
		parameters(0, 0, nullptr)
#else
		toChildren(),
		toOutput(0, 0, nullptr)
#endif
	{
		__debugbreak();
	};


	Module(const Module& n);


	void deepCopy(const Module& n);


	~Module() {};


	void xUpdate_simultaneous();

	void thetaUpdate_simultaneous();

	// Works the same when running on GPU or CPU. The arguments are different, but that is the network's job.
	// float* outActivations, float* outAccumulators are nullptr when on GPU.
	void setArrayPointers(float** ptr_activations, float** ptr_accumulators, float* outActivations, float* outAccumulators);

};
