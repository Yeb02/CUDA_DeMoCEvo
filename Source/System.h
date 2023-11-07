#pragma once


#include "VirtualTrial.h"
#include "Network.h"




struct SystemParameters {

	
	int nBatches;
	int nPerturbations;
	int nEvaluationTrials;
	int nSupervisedTrials;
	int nTeachers;
	int seedSize;
	float potentialChange;


	// learningRate is for the optimizer of the hyper networks, perturbationMagnitude for the
	// norm of the perturbation added to the generated parameters before evaluation.
	// There could be a third hyperparameter, for the magnitude of the perturbations in the "target"
	// given to the hypernetworks for gradient calculations (would be redundant with learningRate were 
	// the loss linear but it is quadratic). But enough hyperparameters already.

	float learningRate;
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

	System(SystemParameters& params, Trial* trial, bool libtorchUsesCuda);
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
		potentialChange = params.potentialChange;

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
	
	// size (2*nPerturbations + 1). Only needed to replace the worst teacher with the best agent at this generation.
	Network** agents;

	int currentNTeachers;
	int bestAgentID;

	std::unique_ptr<GeneratorNode> rootGeneratorNode;

	std::vector<Network*> teachers;

	// a copy of a teacher in the teachers array, because as of now i am unsure if/when to update the teachers weights during its "lessons"
	Network* activeTeacher;

	// size nPerturbations
	float* coefficients;

	// size nEvaluationTrials (* (2*nPerturbations + 1)) (nEvaluationTrials +1 for temp calculations)
	float** agentsScores;
	// size 2*nPerturbations + 1
	float* agentsFitnesses;

	// size nSupervisedTrials (* nTeachers) (nSupervisedTrials +1 for temp calculations)
	float** teachersScores;
	// size nTeachers
	float* teachersFitnesses;


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
	float potentialChange;
};
