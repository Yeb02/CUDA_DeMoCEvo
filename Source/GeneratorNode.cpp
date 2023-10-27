#pragma once

#include "GeneratorNode.h"

int ConnexionGenerator::nPerturbations = 0;


ConnexionGenerator::ConnexionGenerator(int _nRows, int _nCols, int seedSize, float optimizerLR) :
	nRows(_nRows), nCols(_nCols)
{
	int embDim = 3 + (int)(.5f * log2f((float)(nCols * nRows)));

	std::vector<at::Tensor> optimizedParameters;
	
	for (int i = 0; i < N_MATRICES; i++) {
		matrixGenerators[i] = new MatrixGenerator(seedSize, nCols, nRows, embDim);
		std::vector<at::Tensor> params = matrixGenerators[i]->parameters();
		optimizedParameters.insert(optimizedParameters.end(), params.begin(), params.end());

		matrixPerturbations[i] = new Eigen::MatrixXf * [nPerturbations];
		for (int j = 0; j < nPerturbations; j++)
		{
			matrixPerturbations[i][j] = new Eigen::MatrixXf(nRows,nCols);
		}
	}


	for (int i = 0; i < N_VECTORS; i++) {
		vectorGenerators[i] = new VectorGenerator(seedSize, nRows);
		std::vector<at::Tensor> params = vectorGenerators[i]->parameters();
		optimizedParameters.insert(optimizedParameters.end(), params.begin(), params.end());

		vectorPerturbations[i] = new Eigen::VectorXf * [nPerturbations];
		for (int j = 0; j < nPerturbations; j++)
		{
			vectorPerturbations[i][j] = new Eigen::VectorXf(nRows);
		}
	}

	/*
	torch::optim::OptimizerOptions options;
	options.set_lr(lr);
	optimizer->param_groups()[0].set_options(std::make_unique<torch::optim::OptimizerOptions>(options));
	*/ // TODO  (legacy comment ?)

	optimizer = std::make_unique<torch::optim::SGD>(optimizedParameters, optimizerLR);
}


ConnexionGenerator::~ConnexionGenerator()
{
	for (int i = 0; i < N_MATRICES; i++) {
		delete matrixGenerators[i];

		for (int j = 0; j < nPerturbations; j++)
		{
			delete[] matrixPerturbations[i][j];
		}
		delete[] matrixPerturbations[i];
	}
	for (int i = 0; i < N_VECTORS; i++) {
		delete vectorGenerators[i];

		for (int j = 0; j < nPerturbations; j++)
		{
			delete[] vectorPerturbations[i][j];
		}
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
			for (int r = 0; r < nRows; r++) {
				for (int c = 0; c < nCols; c++) {
					(*matrixPerturbations[j][i])(r,c) = NORMAL_01 * perturbationMagnitude;
				}
			}
		}
		for (int j = 0; j < N_VECTORS; j++) {
			for (int r = 0; r < nRows; r++) {
				(*vectorPerturbations[j][i])(r) = NORMAL_01 * perturbationMagnitude;
			}
		}
	}
}


void ConnexionGenerator::createPhenotypeArrays(std::vector<MMatrix>& phenotypeMatrices, std::vector<MVector>& phenotypeVectors, int perturbationID, bool negative)
{
	for (int i = 0; i < N_MATRICES; i++) {
		MMatrix gm(generatedMatrices[i].data_ptr<float>(), nRows, nCols);

		if (perturbationID == -1) {
			phenotypeMatrices[i].noalias() = gm;
			continue;
		}
		if (negative) {
			phenotypeMatrices[i].noalias() = *matrixPerturbations[i][perturbationID] - gm;
		}
		else {
			phenotypeMatrices[i].noalias() = *matrixPerturbations[i][perturbationID] + gm;
		}
		
	}

	for (int i = 0; i < N_VECTORS; i++) 
	{
		MVector gv(generatedVectors[i].data_ptr<float>(), nRows);

		if (perturbationID == -1) {
			phenotypeVectors[i].noalias() = gv;
			continue;
		}
		if (negative) {
			phenotypeVectors[i].noalias() = *vectorPerturbations[i][perturbationID] - gv;
		}
		else {
			phenotypeVectors[i].noalias() = *vectorPerturbations[i][perturbationID] + gv;
		}

		if (i == 1) {
			// applies to (inv) sigmas #ifdef ACTIVATION_VARIANCE. Maps to [0.1,1.0]
			constexpr float f = 1.1f;
			phenotypeVectors[i] = (phenotypeVectors[i].array().tanh() + f) * (1.0f/(f+1.0f)) ;
		}
	}
}


void ConnexionGenerator::accumulateGradient(float* coefficients)
{
	for (int i = 0; i < N_MATRICES; i++) {
		torch::Tensor target = generatedMatrices[i].clone(); 

		MMatrix tm(target.data_ptr<float>(), nRows, nCols);
		for (int j = 0; j < nPerturbations; j++) {
			tm += ((*matrixPerturbations[i][j]).array() * coefficients[j]).matrix();
		}

		torch::Tensor loss = torch::mse_loss(generatedMatrices[i], target);

		loss.backward();
	}

	for (int i = 0; i < N_VECTORS; i++) {
		torch::Tensor target = generatedVectors[i].clone();

		MMatrix tv(target.data_ptr<float>(), nRows);
		for (int j = 0; j < nPerturbations; j++) {
			tv += ((*vectorPerturbations[i][j]).array() * coefficients[j]).matrix();
		}

		torch::Tensor loss = torch::mse_loss(generatedVectors[i], target);

		loss.backward();
	}
}

