#pragma once


#include "TorchNNs.h"
#include "DeMoCEvoCore.h"



struct ConnexionGenerator
{
	static int nPerturbations;

	// matrices are nRows by nCols, vectors are column vectors with nRows;
	int nRows, nCols;

	std::unique_ptr<torch::optim::SGD> optimizer; // SGD ? Adadelta ? TODO

	MatrixGenerator* matrixGenerators[N_MATRICES];
	VectorGenerator* vectorGenerators[N_VECTORS];

	torch::Tensor generatedMatrices[N_MATRICES];
	torch::Tensor generatedVectors[N_VECTORS];

	Eigen::MatrixXf** matrixPerturbations[N_MATRICES];
	Eigen::VectorXf** vectorPerturbations[N_VECTORS];
	
	ConnexionGenerator() {};

	ConnexionGenerator(int _nRows, int _nCols, int seedSize, float optimizerLR);

	~ConnexionGenerator();


	// Generates the arrays (matrices and vectors) with its pytorch networks.
	// As of now, does not support a batch of seeds.
	void generateArrays(torch::Tensor& seed);

	// Fills its perturbations arrays with noise
	void generatePerturbations(float perturbationMagnitude);

	// If perturbationID == -1, no noise is added to the generated arrays. If negative=true, the noise is substracted instead of added.
	void createPhenotypeArrays(std::vector<MMatrix>& phenotypeMatrices, std::vector<MVector>& phenotypeVectors, int perturbationID, bool negative);

	// coefficients correspond to the contribution of each perturbation to the gradient.
	void accumulateGradient(float* coefficients);

	void optimizerStep() { optimizer->step(); }
	void optimizerZero() { optimizer->zero_grad(); }
};



struct GeneratorNode
{
	int inputSize, outputSize;
	int nColumns;
	int nChildren;

	std::vector<ConnexionGenerator> toChildrenGenerators;
	ConnexionGenerator toOutputGenerator;

	std::vector<GeneratorNode> children;

	GeneratorNode() {};

	GeneratorNode(int* inS, int* outS, int* nC, int seedSize, float optimizerLR) :
		nColumns(computeNCols(inS, outS, nC)),
		toOutputGenerator(outS[0], computeNCols(inS, outS, nC), seedSize, optimizerLR),
		inputSize(inS[0]), outputSize(outS[0]), nChildren(nC[0])
	{

		toChildrenGenerators.reserve(nChildren);
		children.reserve(nChildren);
		for (int i = 0; i < nChildren; i++) {
			toChildrenGenerators.emplace_back(inS[1], nColumns, seedSize, optimizerLR);
			children.emplace_back(inS + 1, outS + 1, nC + 1, seedSize, optimizerLR);
		}
	}


	int computeNCols(int* inS, int* outS, int* nC) {
		int cIn = nC[0] > 0 ? outS[1] * nC[0] : 0;
		return inS[0] + cIn;
	}

	void zeroGrad() {

		toOutputGenerator.optimizerZero();

		for (int i = 0; i < nChildren; i++) {
			toChildrenGenerators[i].optimizerZero();
			children[i].zeroGrad();
		}
	}

	void optimizerStep() {

		toOutputGenerator.optimizerStep();

		for (int i = 0; i < nChildren; i++) {
			toChildrenGenerators[i].optimizerStep();
			children[i].optimizerStep();
		}
	}

	void accumulateGrad(float* coefficients) {

		toOutputGenerator.accumulateGradient(coefficients);

		for (int i = 0; i < nChildren; i++) {
			toChildrenGenerators[i].accumulateGradient(coefficients);
			children[i].accumulateGrad(coefficients);
		}
	}
};