#pragma once

#include <memory>
#include "DeMoCEvoCore.h"

#pragma warning( push, 0 )

#include <torch/torch.h>

#pragma warning( pop ) 


struct InternalConnexion { 

	int nRows, nColumns;

	torch::Device* device;

	std::vector<torch::Tensor> matrices;

	// vectors are of size nRows.
	std::vector<torch::Tensor> vectors;


	InternalConnexion(int nRows, int nColumns, torch::Device* _device);

	//Should never be called
	InternalConnexion() {};

	InternalConnexion(const InternalConnexion& gc);

	void deepCopy(const InternalConnexion& gc);

	~InternalConnexion() {};

	void createTensors();

	int getNParameters() {
		return nRows * nColumns * N_MATRICES + nRows * N_VECTORS;
	}

	InternalConnexion(std::ifstream& is);

	void save(std::ofstream& os);
};