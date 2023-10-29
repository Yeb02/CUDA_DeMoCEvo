#pragma once


#include "Network.h"
#include <iostream>



int Network::activationArraySize = 0;
int* Network::inS = nullptr;
int* Network::outS = nullptr;
int* Network::nC = nullptr;
int Network::nLayers = 0;


Network::Network(const Network& pcn)
{
	rootModule.reset(new Module(*(pcn.rootModule.get())));


	if (*device == torch::kCPU) {
		activations = new float[activationArraySize];
		accumulators = new float[activationArraySize];
	}
	else {
		activations = cudaMalloc(*device, activationArraySize);
		accumulators = cudaMalloc(*device, activationArraySize);
	}

	accumulatorsTensor = torch::from_blob(accumulators, { activationArraySize,1 }, torch::TensorOptions().device(*device));
	activationsTensor = torch::from_blob(activations, { activationArraySize,1 }, torch::TensorOptions().device(*device));
	
	activationsTensor = pcn.activationsTensor.clone(); TODO;
	accumulatorsTensor = pcn.accumulatorsTensor.clone(); TODO;


	// The following values will be modified by each node of the phenotype as the pointers are set.
	float* ptr_activations = activations + outS[0];
	float* ptr_accumulators = accumulators + outS[0];
	float* outputActivations = activations;
	float* outputAccumulators = accumulators;

	rootModule->setArrayPointers(
		&ptr_activations,
		&ptr_accumulators,
		outputActivations,
		outputAccumulators
	);

}


void Network::deepCopy(const Network& pcn)
{
	rootModule->deepCopy(*(pcn.rootModule.get()));

	activationsTensor = pcn.activationsTensor.clone(); TODO;
	accumulatorsTensor = pcn.accumulatorsTensor.clone(); TODO;
}


float* Network::getOutput()
{
#ifdef ACTION_L_OBS_O
	return rootModule->inputActivations.data_ptr<float>();
#else
	return rootModule->outputActivations.data_ptr<float>();
#endif
}


Network::Network(GeneratorNode* rootGenerator)
{
	device = rootGenerator->device;

	rootModule = std::make_unique<Module>(*rootGenerator);

	if (*device == torch::kCPU) {
		activations = new float[activationArraySize];
		accumulators = new float[activationArraySize];
	}
	else {
		activations = cudaMalloc(*device,  activationArraySize);
		accumulators = cudaMalloc(*device,  activationArraySize);
	}

	
	accumulatorsTensor = torch::from_blob(accumulators, { activationArraySize,1 }, torch::TensorOptions().device(*device));
	activationsTensor = torch::from_blob(activations, { activationArraySize,1 }, torch::TensorOptions().device(*device));
	activationsTensor.normal_(.0f, .3f);
	
	// The following values will be modified by each node of the phenotype as the pointers are set.
	float* ptr_activations = activations + outS[0];
	float* ptr_accumulators = accumulators + outS[0];
	float* outputActivations = activations;
	float* outputAccumulators = accumulators;

	rootModule->setArrayPointers(
		&ptr_activations,
		&ptr_accumulators,
		outputActivations,
		outputAccumulators
	);

};


void Network::generatePhenotype(GeneratorNode* rootGenerator, int perturbationID, bool negative) {
	activationsTensor.normal_(.0f, .3f);
	//accumulatorsTensor.zero_(); unnecessary, already happens in step().
	rootModule->generatePhenotype(*rootGenerator, perturbationID, negative);
}


void Network::preTrialReset() {
	// zero activations ? probably not.
};


void Network::step(float* input, bool supervised, float* target)
{
	// TODO evolve per module ? If you change it here, it must also be updated in PC_Node_P::xUpdate_simultaneous() 
	constexpr float xlr = .5f;


#ifdef ACTION_L_OBS_O
	MVector vtarget(target, inS[0]);
#endif

#ifdef ACTION_L_OBS_O
	std::copy(input, input + outS[0], rootModule->outputActivations.data());
#else
	std::copy(input, input + inS[0], rootModule->inputActivations.data());
	if (supervised) {
		std::copy(target, target + outS[0], rootModule->outputActivations.data());
	}
#endif



	// inference
	for (int i = 0; i < 10; i++)
	{
		accumulatorsTensor.zero_();
		
#ifdef ACTION_L_OBS_O


		if (supervised) {
			// store -epsilon_in in the root's inputAccumulators.
			// Even if ACTIVATION_VARIANCE is defined, do not multiply by invSigmas. The root handles it.
			rootModule->inputAccumulators = vtarget - rootModule->inputActivations;
		}
		rootModule->xUpdate_simultaneous();
		rootModule->inputActivations += xlr * rootModule->inputAccumulators;

		// TODO outputActivations (i.e. observations) update ? Update mask ? low lr ? ...
#else 


		rootModule->xUpdate_simultaneous();
		if (!supervised) {
			rootModule->outputActivations += xlr * rootModule->outputAccumulators;
		}


#endif
	}

	// learning:
#ifdef MODULATED
	rootModule->thetaUpdate_simultaneous();
#else
	// if the network does not use modulation and is not supervised, there is nothing to learn, so thetaUpdate is pointless.
	if (supervised) {
		rootModule->thetaUpdate_simultaneous();
	}
#endif

}


void Network::save(std::ofstream& os)
{
	int version = 0;
	WRITE_4B(version, os); 

	// TODO. 

}


Network::Network(std::ifstream& is) 
{
	/*int version;
	READ_4B(version, is);*/

	// TODO.
}
