#pragma once

#include <memory>
#include "DeMoCEvoCore.h"


struct InternalConnexion { 

	int nRows, nColumns;

	std::unique_ptr<float[]> storage;

	std::vector<MMatrix> matrices;

	// vectors are of size nRows.
	std::vector<MVector> vectors;


	InternalConnexion(int nRows, int nColumns);

	//Should never be called
	InternalConnexion() {};

	InternalConnexion(const InternalConnexion& gc);

	~InternalConnexion() {};

	int getNParameters() {
		return nRows * nColumns * N_MATRICES + nRows * N_VECTORS;
	}

	void createArraysFromStorage();

	InternalConnexion(std::ifstream& is);

	void save(std::ofstream& os);
};