#pragma once

#include "InternalConnexion.h"

InternalConnexion::InternalConnexion(int nLines, int nColumns) :
	nColumns(nColumns), nLines(nLines)
{
	int s = nLines * nColumns;

	H = std::make_unique<float[]>(s);
	E = std::make_unique<float[]>(s);
	wLifetime = std::make_unique<float[]>(s);

	w = std::make_unique<float[]>(s);
	biases = std::make_unique<float[]>(nLines);

	eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);
	D = std::make_unique<float[]>(s);
	alpha = std::make_unique<float[]>(s);
	gamma = std::make_unique<float[]>(s);

#ifdef OJA
	delta = std::make_unique<float[]>(s);
#endif

#ifdef STDP
	STDP_mu = std::make_unique<float[]>(nLines);
	STDP_lambda = std::make_unique<float[]>(nLines);
#endif

#ifndef	ZERO_WL_BEFORE_TRIAL
	// to have a defined initialisation. ifdef not needed, happens in pretrialUpdate
	zeroWlifetime(); 
#endif
}


#ifdef DROPOUT
void InternalConnexion::dropout() {
	int s = nLines * nColumns;
	if (s == 0) return;

	float normalizator = .3f * powf((float)s, -.5f); // xavier

	SET_BINOMIAL(s, .01f);
	int _nMutations = BINOMIAL;
	for (int i = 0; i < _nMutations; i++) {
		int id = INT_0X(s);

		wLifetime[id] = 0.0f;
		H[id] = 0.0f;
		E[id] = 0.0f;

#ifdef RANDOM_WB
		w[id] = NORMAL_01 * normalizator;
#endif 

	}
}
#endif

void InternalConnexion::zeroEH() {
	int s = nLines * nColumns;
	std::fill(&H[0], &H[s], 0.0f);
	std::fill(&E[0], &E[s], 0.0f);
}


void InternalConnexion::zeroWlifetime()
{
	int s = nLines * nColumns;
	std::fill(&wLifetime[0], &wLifetime[s], 0.0f);
}

#ifdef RANDOM_WB
void InternalConnexion::randomInitWB()
{
	int s = nLines * nColumns;
	if (s == 0) return;
	float normalizator = .3f * powf((float)s, -.5f); // xavier
	

	for (int i = 0; i < s; i++) {
		w[i] = .2f * (UNIFORM_01 - .5f);
		//w[i] = NORMAL_01 * normalizator;
	}

	for (int i = 0; i < nLines; i++) {
		biases[i] = NORMAL_01 * .1f;
		//biases[i] = 0.0f;
	}
}
#endif

