#pragma once


#include "TorchNNs.h"
#include "DeMoCEvoCore.h"



struct ConnexionGenerator
{
	static int nPerturbations;

	// matrices are nRows by nCols, vectors are column vectors with nRows;
	int nRows, nCols;

	torch::optim::SGD* optimizer; // SGD ? Adadelta ? TODO

	// cpu if cuda is not used/available, GPU otherwise.
	torch::Device* device;

	MatrixGenerator* matrixGenerators[N_MATRICES];
	VectorGenerator* vectorGenerators[N_VECTORS];

	torch::Tensor generatedMatrices[N_MATRICES];
	torch::Tensor generatedVectors[N_VECTORS];

	torch::Tensor* matrixPerturbations[N_MATRICES];
	torch::Tensor* vectorPerturbations[N_VECTORS];
	
	ConnexionGenerator() {};

	ConnexionGenerator(int _nRows, int _nCols, int seedSize, float optimizerLR, torch::Device* device);

	~ConnexionGenerator();


	// Generates the arrays (matrices and vectors) with its pytorch networks.
	// As of now, does not support a batch of seeds.
	void generateArrays(torch::Tensor& seed);

	// Fills its perturbations arrays with noise
	void generatePerturbations(float perturbationMagnitude);

	// If perturbationID == -1, no noise is added to the generated arrays. If negative=true, the noise is substracted instead of added.
	void createPhenotypeArrays(std::vector<torch::Tensor>& phenotypeMatrices, std::vector<torch::Tensor>& phenotypeVectors, int perturbationID, bool negative);

	// coefficients correspond to the contribution of each perturbation to the gradient.
	void accumulateGradient(float* coefficients);

	void optimizerStep() { optimizer->step(); }
	void optimizerZero() { optimizer->zero_grad(); }
};



struct GeneratorNode
{
	int inputSize, outputSize;
	int nColumns, nRows;
	int nChildren;

	// cpu if cuda is not used/available, GPU otherwise.
	torch::Device* device;


	std::vector<GeneratorNode> children;

#ifdef ONE_MATRIX
	ConnexionGenerator parametersGenerator;
#else
	std::vector<ConnexionGenerator> toChildrenGenerators;
	ConnexionGenerator toOutputGenerator;
#endif



	GeneratorNode() :
		children() 
#ifdef ONE_MATRIX
#else
		,toChildren()
#endif
	{};

	GeneratorNode(int* inS, int* outS, int* nC, int seedSize, float optimizerLR, torch::Device* _device) :
#ifdef ONE_MATRIX
		parametersGenerator(computeNRows(inS, outS, nC), computeNCols(inS, outS, nC), seedSize, optimizerLR, _device),
#else
		toChildren(),
		toOutputGenerator(outS[0], computeNCols(inS, outS, nC), seedSize, optimizerLR, _device),
#endif
		inputSize(inS[0]), outputSize(outS[0]), nChildren(nC[0]), children(),
		device(_device)
	{
		nColumns = computeNCols(inS, outS, nC);
		nRows = computeNRows(inS, outS, nC);

		children.reserve(nChildren);
		for (int i = 0; i < nChildren; i++) {
			children.emplace_back(inS + 1, outS + 1, nC + 1, seedSize, optimizerLR, device);
		}

#ifndef ONE_MATRIX
		toChildrenGenerators.reserve(nChildren);
		for (int i = 0; i < nChildren; i++) {
			toChildrenGenerators.emplace_back(inS[1], nColumns, seedSize, optimizerLR, device);
		}
#endif
	}


	int computeNCols(int* inS, int* outS, int* nC) {
		int cOut = nC[0] > 0 ? outS[1] * nC[0] : 0;
		return inS[0] + cOut;
	}

	int computeNRows(int* inS, int* outS, int* nC) {
		int cIn = nC[0] > 0 ? inS[1] * nC[0] : 0;
		return outS[0] + cIn;
	}

	void generatePerturbations(float perturbationMagnitude) {
		
		for (int i = 0; i < nChildren; i++) {
			children[i].generatePerturbations(perturbationMagnitude);
		}

#ifdef ONE_MATRIX
		parametersGenerator.generatePerturbations(perturbationMagnitude);
#else
		toOutputGenerator.generatePerturbations(perturbationMagnitude);
		for (int i = 0; i < nChildren; i++) {
			toChildrenGenerators[i].generatePerturbations(perturbationMagnitude);
		}
#endif

	}

	void zeroGrad() {

		for (int i = 0; i < nChildren; i++) {
			children[i].zeroGrad();
		}


#ifdef ONE_MATRIX
		parametersGenerator.optimizerZero();
#else
		toOutputGenerator.optimizerZero();
		for (int i = 0; i < nChildren; i++) {
			toChildrenGenerators[i].optimizerZero();
		}
#endif
	}

	void optimizerStep() {


		for (int i = 0; i < nChildren; i++) {
			children[i].optimizerStep();
		}


#ifdef ONE_MATRIX
		parametersGenerator.optimizerStep();
#else
		toOutputGenerator.optimizerStep();
		for (int i = 0; i < nChildren; i++) {
			toChildrenGenerators[i].optimizerStep();
		}
#endif
	}

	void accumulateGradient(float* coefficients) {

		for (int i = 0; i < nChildren; i++) {
			children[i].accumulateGradient(coefficients);
		}


#ifdef ONE_MATRIX
		parametersGenerator.accumulateGradient(coefficients);
#else
		toOutputGenerator.accumulateGradient(coefficients);
		for (int i = 0; i < nChildren; i++) {
			toChildrenGenerators[i].accumulateGradient(coefficients);
		}
#endif

	}
};