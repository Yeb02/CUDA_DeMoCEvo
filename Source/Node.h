#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "InternalConnexion.h"
#include "GeneratorNode.h"



struct Node {

	int inputSize, outputSize, nChildren;   

	std::vector<Node> children;

	std::vector<InternalConnexion> toChildren;
	InternalConnexion toOutput;


	// Concatenation of this node's input and children's output.
	MVector inCoutActivations;
	MVector inCoutAccumulators;

	// This node's input
	MVector inputActivations;
	MVector inputAccumulators;

	// This node's output.
	MVector outputActivations;
	MVector outputAccumulators;


	Node(int* inS, int* outS, int* nC);

	
	Node(GeneratorNode& generator, int perturbationID, bool negative);

	// Should never be called.
	Node();


	Node(const Node& n);


	~Node() {};


	void xUpdate_simultaneous();

	void thetaUpdate_simultaneous();


	// Works the same when running on GPU or CPU. The arguments are different, but that is the network's job.
	void setArrayPointers(float** ptr_activations, float** ptr_accumulators, float* outActivations, float* outAccumulators);

	// stupid but thats the only way to do complex operations in an initializer list.
	static int computeNCols(int* inS, int* outS, int* nC) {
		int cIn = nC[0] > 0 ? outS[1] * nC[0] : 0;
		return inS[0] + cIn;
	}
};
