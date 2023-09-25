#pragma once

#include <memory>
#include "config.h"
#include "Random.h"


struct InternalConnexion { 

	int nLines, nColumns;

	// constant quantities:
	std::unique_ptr<float[]> A;
	std::unique_ptr<float[]> B;
	std::unique_ptr<float[]> C;
	std::unique_ptr<float[]> D;
	std::unique_ptr<float[]> eta;	// in [0, 1]
	std::unique_ptr<float[]> alpha;
	std::unique_ptr<float[]> gamma; // in [0, 1]

#ifdef OJA
	std::unique_ptr<float[]> delta; // in [0, 1]
#endif

	


#ifdef STDP
	std::unique_ptr<float[]> STDP_mu;
	std::unique_ptr<float[]> STDP_lambda;
#endif

	// variable quantities:

	std::unique_ptr<float[]> H;
	std::unique_ptr<float[]> E;
	// Initialized to 0 when and only when the connexion is created. 
	std::unique_ptr<float[]> wLifetime;


	// If RANDOM_WB, reset to random values at the beginning of each trial.
	// Otherwise constant.
	std::unique_ptr<float[]> w;
	std::unique_ptr<float[]> biases;


	InternalConnexion(int nLines, int nColumns);
	//Should never be called
	InternalConnexion() {};
	~InternalConnexion() {};

#ifdef RANDOM_WB
	void randomInitWB();
#endif

#ifdef DROPOUT
	void dropout();
#endif

	void zeroEH();

	// only called at construction.
	void zeroWlifetime();

};