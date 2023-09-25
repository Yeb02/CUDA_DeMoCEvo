#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "Random.h"
#include "InternalConnexion.h"
#include "config.h"



// Util:
inline int binarySearch(std::vector<float>& proba, float value) {
	int inf = 0;
	int sup = (int)proba.size() - 1;

	if (proba[inf] > value) {
		return inf;
	}

	int mid;
	int max_iter = 15;
	while (sup - inf >= 1 && max_iter--) {
		mid = (sup + inf) / 2;
		if (proba[mid] < value && value <= proba[mid + 1]) {
			return mid + 1;
		}
		else if (proba[mid] < value) {
			inf = mid;
		}
		else {
			sup = mid;
		}
	}
	return 0; // not necessarily a failure, since floating point approximation prevents the sum from reaching 1.
	//throw "Binary search failure !";
}

inline int binarySearch(float* proba, float value, int size) {
	int inf = 0;
	int sup = size - 1;

	if (proba[inf] > value) {
		return inf;
	}

	int mid;
	int max_iter = 15;
	while (sup - inf >= 1 && max_iter--) {
		mid = (sup + inf) / 2;
		if (proba[mid] < value && value <= proba[mid + 1]) {
			return mid + 1;
		}
		else if (proba[mid] < value) {
			inf = mid;
		}
		else {
			sup = mid;
		}
	}
	return 0; // not necessarily a failure, since floating point approximation prevents the sum from reaching 1.
	//throw "Binary search failure !";
}

struct Node {

	int inputSize, outputSize; 

	float totalM[MODULATION_VECTOR_SIZE]; // parent's + local M.    

	std::vector<Node> children;

	InternalConnexion toChildren;
	InternalConnexion toModulation;
	InternalConnexion toOutput;

	
	// Used as the multiplied vector in matrix operations. Layout:
	// input -> modulation.out -> complexChildren.out -> memoryChildren.out
	float* inputArray;

	// Used as the result vector in matrix operations. Layout:
	// output -> modulation.in -> complexChildren.in -> memoryChildren.in
	float* destinationArray;

#ifdef STDP
	// Same layout as PreSynActs, i.e.
	// output -> modulation.in -> complexChildren.in -> memoryChildren.in
	float* accumulatedPreSynActs;
#endif

#ifdef SATURATION_PENALIZING
	// Layout:
	// Modulation -> (complexChildren->inputSize) -> (memoryChildren->inputSize (mn owns it))
	float* averageActivation;

	// A parent updates it for its children (in and out), not for itself.
	float* globalSaturationAccumulator;
#endif

	// Should never be called.
	Node();

	// Should never be called.
	Node(const Node&) { 
		__debugbreak(); 
	}

	Node(int* inS, int* outS, int* nC);

	// stupid but thats the only way to do complex operations in an initializer list.
	static int computeNCols(int* inS, int* outS, int nC) {
		int cIn = nC > 0 ? outS[1] * nC : 0;
		return inS[0] + MODULATION_VECTOR_SIZE + cIn;
	}

	~Node() {};


	void preTrialReset();

	void forward();


	// The last 2 parameters are optional :
	// - aa only used when SATURATION_PENALIZING is defined
	// - acc_pre_syn_acts only used when STDP is defined
	void setArrayPointers(float** pre_syn_acts, float** post_syn_acts, float** aa, float** acc_pre_syn_acts);

#ifdef SATURATION_PENALIZING
	void setglobalSaturationAccumulator(float* globalSaturationAccumulator);
#endif
};
