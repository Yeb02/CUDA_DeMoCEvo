#pragma once

#include "InternalConnexion.h"


InternalConnexion::InternalConnexion(int nRows, int nColumns, torch::Device* _device) :
	nColumns(nColumns), nRows(nRows), device(_device)
{
	createTensors();
}


InternalConnexion::InternalConnexion(const InternalConnexion& gc)
{
	nRows = gc.nRows;
	nColumns = gc.nColumns;
	device = gc.device;

	createTensors();

	deepCopy(gc);
}


void InternalConnexion::deepCopy(const InternalConnexion& gc) {
	matrices.resize(N_MATRICES);
	for (int i = 0; i < N_MATRICES; i++)
	{
		matrices[i] = gc.matrices[i].clone();  // TODO check that it happens in the preallocated memory.
	}

	vectors.resize(N_VECTORS);
	for (int i = 0; i < N_VECTORS; i++)
	{
		vectors[i] = gc.vectors[i].clone(); // TODO check that it happens in the preallocated memory.
	}
}


void InternalConnexion::createTensors()
{
	matrices.resize(N_MATRICES);
	for (int i = 0; i < N_MATRICES; i++)
	{
		matrices[i] = torch::zeros(torch::IntArrayRef{ nRows, nColumns }, torch::TensorOptions().dtype(torch::kFloat32).device(*device));
	}

	vectors.resize(N_VECTORS);
	for (int i = 0; i < N_VECTORS; i++)
	{
		vectors[i] = torch::zeros(torch::IntArrayRef{ nRows, 1 }, torch::TensorOptions().dtype(torch::kFloat32).device(*device));
	}
}


InternalConnexion::InternalConnexion(std::ifstream& is)
{
	READ_4B(nRows, is);
	READ_4B(nColumns, is);
}


void InternalConnexion::save(std::ofstream& os)
{
	WRITE_4B(nRows, os);
	WRITE_4B(nColumns, os);
}