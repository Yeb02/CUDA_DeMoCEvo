#pragma once

#include "InternalConnexion.h"


InternalConnexion::InternalConnexion(int nRows, int nColumns) :
	nColumns(nColumns), nRows(nRows)
{
	storage = std::make_unique<float[]>(getNParameters());

	createArraysFromStorage();
}


InternalConnexion::InternalConnexion(const InternalConnexion& gc)
{
	nRows = gc.nRows;
	nColumns = gc.nColumns;

	int s = getNParameters();

	storage = std::make_unique<float[]>(s);
	std::copy(gc.storage.get(), gc.storage.get() + s, storage.get());

	createArraysFromStorage();
}


void InternalConnexion::createArraysFromStorage()
{
	int s = nRows * nColumns;

	float* _storagePtr = storage.get();

	matrices.reserve(N_MATRICES);
	for (int i = 0; i < N_MATRICES; i++)
	{
		matrices.emplace_back(_storagePtr, nRows, nColumns);
		_storagePtr += s;
	}

	vectors.reserve(N_VECTORS);
	for (int i = 0; i < N_VECTORS; i++)
	{
		vectors.emplace_back(_storagePtr, nRows);
		_storagePtr += nRows;
	}
}


InternalConnexion::InternalConnexion(std::ifstream& is)
{
	READ_4B(nRows, is);
	READ_4B(nColumns, is);

	storage = std::make_unique<float[]>(getNParameters());
	is.read(reinterpret_cast<char*>(storage.get()), getNParameters() * sizeof(float));

	createArraysFromStorage();
}


void InternalConnexion::save(std::ofstream& os)
{
	WRITE_4B(nRows, os);
	WRITE_4B(nColumns, os);

	os.write(reinterpret_cast<const char*>(storage.get()), getNParameters() * sizeof(float));
}