#pragma once


#include "Trial.h"
#include "Network.h"
#include "DeMoCEvoCore.h"




struct SystemParameters {

	float learningRate;
	int nBatches;
	int nPerturbations;
	int nEvaluationTrials;
	int nSupervisedTrials;
	int nTeachers;
	int seedSize;
	float perturbationMagnitude;

	int* inSizePerL;
	int* outSizePerL;
	int* nChildrenPerL;
	int treeDepth;

	SystemParameters() {};
};



class System
{

public:

	System(SystemParameters& params, Trial* trial);
	~System() {};

	void setParameters(SystemParameters& params)
	{
		learningRate = params.learningRate;
		nPerturbations = params.nPerturbations;
		nEvaluationTrials = params.nEvaluationTrials;
		nSupervisedTrials = params.nSupervisedTrials;
		nTeachers = params.nTeachers;
		nBatches = params.nBatches;
		seedSize = params.seedSize;
		perturbationMagnitude = params.perturbationMagnitude;

		inSizePerL = params.inSizePerL;
		outSizePerL = params.outSizePerL;
		nChildrenPerL = params.nChildrenPerL;
		treeDepth = params.treeDepth;

	};


	void evolve(int nSteps);


private:

	void teachAndEvaluate();
	void computeCoefficients();
	void updateTeachers();


	Trial* trial;

	std::unique_ptr<GeneratorNode> rootGeneratorNode;

	std::vector<Network*> teachers;

	// size nPerturbations
	float* coefficients;

	// size nEvaluationTrials * (2*nPerturbations + 1)
	float** agentsScores;



	int* inSizePerL;
	int* outSizePerL;
	int* nChildrenPerL;
	int treeDepth;

	float learningRate;
	int nBatches;
	int nPerturbations;
	int nEvaluationTrials;
	int nSupervisedTrials;
	int nTeachers;
	int seedSize;
	float perturbationMagnitude;
};
