#pragma once


#include "TorchNNs.h"
#include "Trial.h"
#include "Network.h"
#include "config.h"

struct GeneratorParameters {

	float lr;
	int nNetsPerBatch;
	int nTrialsPerNet;
	int gaussianVecSize;
	float elitePercentage;

	// per elite specimen !
	int nUpdatedPoints;
	float updateRadius;

	//defaults:
	GeneratorParameters()
	{
		lr = .01f;
		nNetsPerBatch = 30;
		nTrialsPerNet = 40;
		gaussianVecSize = 30;
		elitePercentage = .1f;
		nUpdatedPoints = 10;
		updateRadius = .2f;

#ifdef _DEBUG
		nNetsPerBatch = 5;
		nTrialsPerNet = 4;
		gaussianVecSize = 5;
		elitePercentage = .5f;
		nUpdatedPoints = 2;
		updateRadius = .2f;
#endif
	}
};



struct Generator 
{
	Generator(int netInSize, int netOutSize);
	~Generator() {};

	void setParameters(GeneratorParameters& params) 
	{
		lr = params.lr;
		nNetsPerBatch = params.nNetsPerBatch;
		nTrialsPerNet = params.nTrialsPerNet;
		gaussianVecSize = params.gaussianVecSize;
		elitePercentage = params.elitePercentage;
		nUpdatedPoints = params.nUpdatedPoints;
		updateRadius = params.updateRadius;
	};

	void step(Trial* trial);

	// NO_GRAD
	Network* createNet();

	// to unbloat createNet(). kinda useless as it happens only once.
	void fillNode(Node* node, int layer, float* seed);

	// fills nodes with n's nodes, breadth first traversal
	void flattenNetwork(Node** nodes, Network* n);

	// Forward pass at the given layer on (Xs,NN(X0)), while accumulating gradients.
	void accumulateGrads(int layer, float* Xs, float* X0);

	// Given a network (whose phenotype has been deleted), generates seeds
	// close to those of the network and SGD on these to have the same output
	// as the Net's params
	void examplifyNet(Network* n);

	void save();
	void save1Net();

	
	int nLayers;

	// at indice i, the number of nodes of layer i.
	// Powers of 2, increasing, first element = 1 (topNode)
	std::unique_ptr<int[]> nNodesAtLayer;

	// at indice i, the number of children of nodes of layer i.
	std::unique_ptr<int[]> nChildrenAtLayer;

	// at indice i, the common input size of nodes of layer i.
	std::unique_ptr<int[]> nodesInSizeAtLayer;

	// at indice i, the common output size of nodes of layer i.
	std::unique_ptr<int[]> nodesOutSizeAtLayer;

	int seedSize;

	std::vector<Matrixator> matrixators;    // size nLayers
	std::vector<Specialist> matSpecialists; // size N_MATRICES * nLayers
	std::vector<Specialist> arrSpecialists; // size N_ARRAYS * nLayers (so 0 if N_ARRAYS == 0)


	std::unique_ptr<torch::optim::SGD> optimizer; // different lrs per layer ? Adadelta ? TODO


	std::vector<int> nLines; 
	std::vector<int> nCols;

	std::vector<int> colEmbS;
	std::vector<int> lineEmbS;


	int netInSize, netOutSize;

	
	float lr;
	int gaussianVecSize;
	int nNetsPerBatch;
	int nTrialsPerNet;
	float elitePercentage;
	int nUpdatedPoints;
	float updateRadius;
};
