#pragma once


#include "Network.h"
#include <iostream>



int Network::activationArraySize = 0;
int* Network::inS = nullptr;
int* Network::outS = nullptr;
int* Network::nC = nullptr;
int Network::nLayers = 0;



Network::Network()
{
	// Quantites created in createdPhenotype:
	rootNode.reset(NULL);
	activations.reset(NULL);
	accumulators.reset(NULL);
}


Network::Network(const Network& pcn)
	
{

	rootNode.reset(new Node(*(pcn.rootNode.get())));

	activations = std::make_unique<float[]>(activationArraySize);

	// if changed here, change createPhenotype too.
	//std::fill(activations.get(), activations.get() + activationArraySize, 0.0f);
	for (int i = 0; i < activationArraySize; i++) {
		activations[i] = NORMAL_01 * .3f;
	}

	accumulators = std::make_unique<float[]>(activationArraySize);

	// The following values will be modified by each node of the phenotype as the pointers are set.
	float* ptr_activations = activations.get() + outS[0];
	float* ptr_accumulators = accumulators.get() + outS[0];
	float* outputActivations = activations.get();
	float* outputAccumulators = accumulators.get();

	rootNode->setArrayPointers(
		&ptr_activations,
		&ptr_accumulators,
		outputActivations,
		outputAccumulators
	);

	setInitialActivations(pcn.activations.get());
}


float* Network::getOutput()
{
#ifdef ACTION_L_OBS_O
	return rootNode->inputActivations.data();
#else
	return rootNode->outputActivations.data();
#endif
}


void Network::destroyPhenotype() {
	rootNode.reset(NULL);

	activations.reset(NULL);
	accumulators.reset(NULL);
}


void Network::createPhenotype(GeneratorNode* rootGenerator, int perturbationID, bool negative)
{

	if (rootNode.get() != NULL)
	{
		std::cerr << "Called createPhenotype on a Network that already had a phenotype !" << std::endl;
		return;
	}


	rootNode.reset(new Node(*rootGenerator, perturbationID, negative));


	activations = std::make_unique<float[]>(activationArraySize);

	// if changed here, change (teacher) copy constructor too.
	//std::fill(activations.get(), activations.get() + activationArraySize, 0.0f);
	for (int i = 0; i < activationArraySize; i++) {
		activations[i] = NORMAL_01 * .3f;
	}

	accumulators = std::make_unique<float[]>(activationArraySize);

	// The following values will be modified by each node of the phenotype as the pointers are set.
	float* ptr_activations = activations.get() + outS[0];
	float* ptr_accumulators = accumulators.get() + outS[0];
	float* outputActivations = activations.get();
	float* outputAccumulators = accumulators.get();

	rootNode->setArrayPointers(
		&ptr_activations,
		&ptr_accumulators,
		outputActivations,
		outputAccumulators
	);

};


void Network::preTrialReset() {
	//std::fill(activations.get(), activations.get() + activationArraySize, 0.0f); // ?
};


void Network::step(float* input, bool supervised, float* target)
{
	// TODO evolve per module ? If you change it here, it must also be updated in PC_Node_P::xUpdate_simultaneous() 
	constexpr float xlr = .5f;


#ifdef ACTION_L_OBS_O
	MVector vtarget(target, inS[0]);
#endif

#ifdef ACTION_L_OBS_O
	std::copy(input, input + outS[0], rootNode->outputActivations.data());
#else
	std::copy(input, input + inS[0], rootNode->inputActivations.data());
	if (supervised) {
		std::copy(target, target + outS[0], rootNode->outputActivations.data());
	}
#endif



	// inference
	for (int i = 0; i < 10; i++)
	{
		std::fill(accumulators.get(), accumulators.get() + activationArraySize, 0.0f);


#ifdef ACTION_L_OBS_O


		if (supervised) {
			// store -epsilon_in in the root's inputAccumulators.
			// Even if ACTIVATION_VARIANCE is defined, do not multiply by invSigmas. The root handles it.
			rootNode->inputAccumulators = vtarget - rootNode->inputActivations;
		}
		rootNode->xUpdate_simultaneous();
		rootNode->inputActivations += xlr * rootNode->inputAccumulators;

		// TODO outputActivations (i.e. observations) update ? Update mask ? low lr ? ...
#else 


		rootNode->xUpdate_simultaneous();
		if (!supervised) {
			rootNode->outputActivations += xlr * rootNode->outputAccumulators;
		}


#endif
	}

	// learning


#ifdef MODULATED
	rootNode->thetaUpdate_simultaneous();
#else
	// if the network does not use modulation and is not supervised, there is nothing to learn, so thetaUpdate is pointless.
	if (supervised) {
		rootNode->thetaUpdate_simultaneous();
	}
#endif

}


void Network::save(std::ofstream& os)
{
	int version = 0;
	WRITE_4B(version, os); // version

	// TODO. Write the phenotypic parameters and references to the genotypic ones.

}


Network::Network(std::ifstream& is) 
{
	/*int version;
	READ_4B(version, is);*/

	// TODO.
}
