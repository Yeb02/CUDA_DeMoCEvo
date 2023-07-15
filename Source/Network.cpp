#pragma once

#include "Network.h"
#include <iostream>


// Everything is set up externally
Network::Network(int inputSize, int outputSize, float* seed) :
	inputSize(inputSize), outputSize(outputSize), seed(seed)
{
	nInferencesOverLifetime = 0;
	topNode.reset(NULL);
}


float* Network::getOutput() {
	return topNode->preSynActs;
}


void Network::postTrialUpdate() {

}


void Network::destroyPhenotype() {
	topNode.reset(NULL);
	postSynActs.reset(NULL);
	preSynActs.reset(NULL);
#ifdef SATURATION_PENALIZING
	averageActivation.reset(NULL);
#endif
#ifdef STDP
	accumulatedPreSynActs.reset(NULL);
#endif

}


void Network::createPhenotype(int* inS, int* outS, int* nC, int*nN, int nL) {
	if (topNode.get() == NULL) 
	{
		topNode.reset(new Node(inS, outS, nC));

		preSynActsArraySize = 0;
		postSynActArraySize = 0;
#ifdef SATURATION_PENALIZING
		averageActivationArraySize = 0;
#endif
		for (int i = 0; i < nL; i++) 
		{
			int cIs = nC[i] == 0 ? 0 : inS[i+1];
			preSynActsArraySize += nN[i] *
				(outS[i] + MODULATION_VECTOR_SIZE + cIs * nC[i]);

			int cOs = nC[i] == 0 ? 0 : outS[i + 1];
			postSynActArraySize += nN[i] *
				(inS[i] + MODULATION_VECTOR_SIZE + cOs * nC[i]);

#ifdef SATURATION_PENALIZING
			averageActivationArraySize += nN[i] *
				(MODULATION_VECTOR_SIZE + cIs * nC[i]);
#endif
		}

		preSynActs = std::make_unique<float[]>(preSynActsArraySize);
		postSynActs = std::make_unique<float[]>(postSynActArraySize);

		float* ptr_accumulatedPreSynActs = nullptr;
#ifdef STDP
		accumulatedPreSynActs = std::make_unique<float[]>(preSynActsArraySize);
		ptr_accumulatedPreSynActs = accumulatedPreSynActs.get();
#endif

		float* ptr_averageActivation = nullptr;
#ifdef SATURATION_PENALIZING
		averageActivation = std::make_unique<float[]>(averageActivationArraySize);
		ptr_averageActivation = averageActivation.get();

		saturationPenalization = 0.0f;
		topNode->setglobalSaturationAccumulator(&saturationPenalization);
		std::fill(averageActivation.get(), averageActivation.get() + averageActivationArraySize, 0.0f);
#endif

		
		// The following values will be modified by each node of the phenotype as the pointers are set.
		float* ptr_postSynActs = postSynActs.get();
		float* ptr_preSynActs = preSynActs.get();
		topNode->setArrayPointers(
			&ptr_postSynActs,
			&ptr_preSynActs,
			&ptr_averageActivation,
			&ptr_accumulatedPreSynActs
		);

		nInferencesOverLifetime = 0;
	}
};


void Network::preTrialReset() {

	std::fill(postSynActs.get(), postSynActs.get() + postSynActArraySize, 0.0f);
	//std::fill(preSynActs.get(), preSynActs.get() + preSynActsArraySize, 0.0f); // is already set to the biases.
#ifdef STDP
	std::fill(accumulatedPreSynActs.get(), accumulatedPreSynActs.get() + preSynActsArraySize, 0.0f);
#endif
	
	topNode->preTrialReset();
};


void Network::step(const std::vector<float>& obs) 
{
	nInferencesOverLifetime++;
	std::copy(obs.begin(), obs.end(), topNode->postSynActs);
	std::fill(topNode->totalM, topNode->totalM + MODULATION_VECTOR_SIZE, 0.0f);
	topNode->forward();
}


#ifdef SATURATION_PENALIZING
float Network::getSaturationPenalization()
{

	float p1 = averageActivationArraySize != 0 ? saturationPenalization / (nInferencesOverLifetime * averageActivationArraySize) : 0.0f;


	float p2 = 0.0f;
	float invNInferencesN = 1.0f / nInferencesOverLifetime;
	for (int i = 0; i < averageActivationArraySize; i++) {
		p2 += powf(abs(averageActivation[i]) * invNInferencesN, 6.0f);
	}
	p2 /= (float) averageActivationArraySize;
	


	constexpr float µ = .5f;
	return µ * p1 + (1 - µ) * p2;
}
#endif


void Network::save(std::ofstream& os)
{
	int version = 0;
	WRITE_4B(version, os); // version

	WRITE_4B(inputSize, os);
	WRITE_4B(outputSize, os);

	// TODO seeds.
}

Network::Network(std::ifstream& is)
{
	int version;
	READ_4B(version, is);
	
	READ_4B(inputSize, is);
	READ_4B(outputSize, is);

	// TODO seeds. Fill an array used externally by generators

	topNode.reset(NULL);
}