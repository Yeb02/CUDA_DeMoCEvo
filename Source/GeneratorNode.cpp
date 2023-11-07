#pragma once

#include "GeneratorNode.h"

int ConnexionGenerator::nPerturbations = 0;


ConnexionGenerator::ConnexionGenerator(int _nRows, int _nCols, int seedSize, float optimizerLR, torch::Device* _device) :
	nRows(_nRows), nCols(_nCols)
{
	device = _device;

	int embDim = 3 + (int)(.5f * log2f((float)(nCols * nRows)));

	std::vector<at::Tensor> optimizedParameters;
	
	for (int i = 0; i < N_MATRICES; i++) {
		matrixGenerators[i] = new MatrixGenerator(seedSize, nCols, nRows, embDim);
		matrixGenerators[i]->to(*device);
		
		std::vector<at::Tensor> params = matrixGenerators[i]->parameters();
		optimizedParameters.insert(optimizedParameters.end(), params.begin(), params.end());

		matrixPerturbations[i] = new torch::Tensor[nPerturbations];
		for (int j = 0; j < nPerturbations; j++)
		{
			matrixPerturbations[i][j] = torch::zeros(torch::IntArrayRef{ nRows, nCols },
				torch::TensorOptions().dtype(torch::kFloat32).device(*device));
		}
	}


	for (int i = 0; i < N_VECTORS; i++) {
		vectorGenerators[i] = new VectorGenerator(seedSize, nRows);
		std::vector<at::Tensor> params = vectorGenerators[i]->parameters();
		optimizedParameters.insert(optimizedParameters.end(), params.begin(), params.end());

		vectorPerturbations[i] = new torch::Tensor[nPerturbations];
		for (int j = 0; j < nPerturbations; j++)
		{
			vectorPerturbations[i][j] = torch::zeros(torch::IntArrayRef{ nRows, 1 },
				torch::TensorOptions().dtype(torch::kFloat32).device(*device));
		}
	}

	/*
	torch::optim::OptimizerOptions options;
	options.set_lr(lr);
	optimizer->param_groups()[0].set_options(std::make_unique<torch::optim::OptimizerOptions>(options));
	*/ // TODO  (legacy comment ?)

	optimizer = new torch::optim::SGD(optimizedParameters, optimizerLR);
}


ConnexionGenerator::~ConnexionGenerator()
{
	for (int i = 0; i < N_MATRICES; i++) {
		delete matrixGenerators[i];

		delete[] matrixPerturbations[i];
	}
	for (int i = 0; i < N_VECTORS; i++) {
		delete vectorGenerators[i];

		delete[] vectorPerturbations[i];
	}
}


void ConnexionGenerator::generateArrays(torch::Tensor& seed)
{
	int s = nRows * nCols;
	for (int i = 0; i < N_MATRICES; i++) {
		generatedMatrices[i] = matrixGenerators[i]->forward(seed);
	}
	s = nRows;
	for (int i = 0; i < N_VECTORS; i++) {
		generatedVectors[i] = vectorGenerators[i]->forward(seed);
	}
}


void ConnexionGenerator::generatePerturbations(float perturbationMagnitude)
{
	for (int i = 0; i < nPerturbations; i++)
	{
		for (int j = 0; j < N_MATRICES; j++) {
			matrixPerturbations[j][i].normal_(0.0f, perturbationMagnitude);
		}
		for (int j = 0; j < N_VECTORS; j++) {
			vectorPerturbations[j][i].normal_(0.0f, perturbationMagnitude);
		}
	}
}


void ConnexionGenerator::createPhenotypeArrays(std::vector<torch::Tensor>& phenotypeMatrices, std::vector<torch::Tensor>& phenotypeVectors, int perturbationID, bool negative)
{

	for (int i = 0; i < N_MATRICES; i++) {

		if (perturbationID == -1) {
			phenotypeMatrices[i] = generatedMatrices[i].detach().clone();
			continue;
		}
		if (negative) {
			phenotypeMatrices[i] = generatedMatrices[i].detach().clone() - matrixPerturbations[i][perturbationID];
		}
		else {
			phenotypeMatrices[i] = generatedMatrices[i].detach().clone() + matrixPerturbations[i][perturbationID];
		}
		
	}

	for (int i = 0; i < N_VECTORS; i++) 
	{
		if (perturbationID == -1) {
			phenotypeVectors[i] = generatedVectors[i].detach().clone();
			continue;
		}
		if (negative) {
			phenotypeVectors[i] = generatedVectors[i].detach().clone() - vectorPerturbations[i][perturbationID];
		}
		else {
			phenotypeVectors[i] = generatedVectors[i].detach().clone() + vectorPerturbations[i][perturbationID];
		}

		if (i == 1) {
			// applies to (inv) sigmas #ifdef ACTIVATION_VARIANCE. Maps to [0.1,1.0]
			constexpr float f = .1f;
			phenotypeVectors[i] = torch::sigmoid(phenotypeVectors[i]) * (1.0-f) + f;
		}
	}
}


void ConnexionGenerator::accumulateGradient(float* coefficients)
{
	for (int i = 0; i < N_MATRICES; i++) {
		torch::Tensor target = generatedMatrices[i].detach().clone();

		for (int j = 0; j < nPerturbations; j++) {
			target += matrixPerturbations[i][j] * coefficients[j];
		}

		torch::Tensor loss = torch::mse_loss(generatedMatrices[i], target);

		loss.backward();
	}

	for (int i = 0; i < N_VECTORS; i++) {
		torch::Tensor target = generatedVectors[i].detach().clone();

		for (int j = 0; j < nPerturbations; j++) {
			target += vectorPerturbations[i][j] * coefficients[j];
		}

		torch::Tensor loss = torch::mse_loss(generatedVectors[i], target);

		loss.backward();
	}
}

