#pragma once

#include "Generator.h"

// src is unchanged.
void normalizeArray(float* src, float* dst, int size) {
	float avg = 0.0f;
	for (int i = 0; i < size; i++) {
		avg += src[i];
	}
	avg /= (float)size;
	float variance = 0.0f;
	for (int i = 0; i < size; i++) {
		dst[i] = src[i] - avg;
		variance += dst[i] * dst[i];
	}
	if (variance < .001f) return;
	float InvStddev = 1.0f / sqrtf(variance / (float)size);
	for (int i = 0; i < size; i++) {
		dst[i] *= InvStddev;
	}
}

// src is unchanged. 
void rankArray(float* src, std::vector<int>& dst, int size) {
	for (int i = 0; i < size; i++) {
		dst[i] = i;
	}
	// sort dst in decreasing order.
	std::sort(dst.begin(), dst.end(), [src](int a, int b) -> bool
		{
			return src[a] > src[b];
		}
	);

	return;
}



Generator::Generator(int netInSize, int netOutSize) :
	netInSize(netInSize), netOutSize(netOutSize)
{ 

	GeneratorParameters params; 
	setParameters(params); 

	std::vector<at::Tensor> optimizerParams;

	nLayers = 4;

	nNodesAtLayer = std::make_unique<int[]>(nLayers);
	nChildrenAtLayer = std::make_unique<int[]>(nLayers);
	nodesInSizeAtLayer = std::make_unique<int[]>(nLayers);
	nodesOutSizeAtLayer = std::make_unique<int[]>(nLayers);

	int a1[4] = {1, 2, 4, 4};
	seedSize = 0;
	for (int i = 0; i < nLayers; i++) {
		nNodesAtLayer[i] = a1[i];
		seedSize += a1[i];
	}
	seedSize *= gaussianVecSize;

	for (int i = 0; i < nLayers-1; i++) {
		nChildrenAtLayer[i] = a1[i+1]/a1[i];
	}
	nChildrenAtLayer[nLayers - 1] = 0;
		
	int a2[4] = { netInSize, 8, 4, 2 };
	for (int i = 0; i < nLayers; i++) {
		nodesInSizeAtLayer[i] = a2[i];
	}

	int a3[4] = { netOutSize, 9, 5, 3 };
	for (int i = 0; i < nLayers; i++) {
		nodesOutSizeAtLayer[i] = a3[i];
	}

	nLines.reserve(nLayers);
	nCols.reserve(nLayers);
	colEmbS.reserve(nLayers);
	lineEmbS.reserve(nLayers);

	matrixators.reserve(nLayers);
	matSpecialists.reserve(nLayers * N_MATRICES);
	arrSpecialists.reserve(nLayers * N_ARRAYS);

	for (int i = 0; i < nLayers; i++)
	{
		
		int cIs = nChildrenAtLayer[i] == 0 ? 0 : nChildrenAtLayer[i] * nodesInSizeAtLayer[i+1];
		nLines.push_back(nodesOutSizeAtLayer[i] + MODULATION_VECTOR_SIZE + cIs);

		int cOs = nChildrenAtLayer[i] == 0 ? 0 : nChildrenAtLayer[i] * nodesOutSizeAtLayer[i+1];
		nCols.push_back(nodesInSizeAtLayer[i] + MODULATION_VECTOR_SIZE + cOs);

		colEmbS.push_back(3 + (int)sqrtf((float)nCols[i])); 
		lineEmbS.push_back(3 + (int)sqrtf((float)nLines[i])); 

		matrixators.emplace_back(gaussianVecSize, nCols[i], nLines[i], colEmbS[i], lineEmbS[i]);

		int in, out;
		in = colEmbS[i] * nCols[i] + lineEmbS[i];
		out = nCols[i];
		// in this order : A,B, C, D, eta, alpha, gamma?, w?, delta?.
		for (int j = 0; j < N_MATRICES; j++) 
		{
			matSpecialists.emplace_back(in, out);
			std::vector<at::Tensor> params = matSpecialists[i* N_MATRICES+j].parameters();
			optimizerParams.insert(optimizerParams.end(), params.begin(), params.end());
		}


		in = lineEmbS[i] * nLines[i] + colEmbS[i] * nCols[i];
		out = nLines[i];
		// in this order: biases?, mu?, lambda?
		for (int j = 0; j < N_ARRAYS; j++)
		{
			arrSpecialists.emplace_back(in, out);
			std::vector<at::Tensor> params = arrSpecialists[i* N_ARRAYS+j].parameters();
			optimizerParams.insert(optimizerParams.end(), params.begin(), params.end());
		}

	}

	optimizer = std::make_unique<torch::optim::SGD>(optimizerParams, lr);
}



void Generator::step(Trial* trial) 
{
	std::vector<float> rawScores;
	std::vector<int> ranks;
	rawScores.resize(nNetsPerBatch);
	ranks.resize(nNetsPerBatch);

	optimizer->zero_grad();

	Network** nets = new Network*[nNetsPerBatch];

	// Create and evaluate nNetsPerBatch networks. Phenotypes are destroyed
	float avgS = 0.0f, maxS = -1000.0f;
	for (int i = 0; i < nNetsPerBatch; i++) {

		nets[i] = createNet(); // creates the phenotype.

		rawScores[i] = 0.0f;

		for (int j = 0; j < nTrialsPerNet; j++) {

			nets[i]->preTrialReset();
			trial->reset(false);

			while (!trial->isTrialOver) {
				nets[i]->step(trial->observations);
				trial->step(nets[i]->getOutput());
			}

			nets[i]->postTrialUpdate();
			rawScores[i] += trial->score;
		}

		//rawScores[i] = nets[i]->topNode->toModulation.alpha[0];

		if (rawScores[i] > maxS) maxS = rawScores[i];
		avgS += rawScores[i];

		nets[i]->destroyPhenotype();
	}

	maxS /= (float)nTrialsPerNet;
	//avgS /= (float)nNetsPerBatch;
	avgS /= (float)(nTrialsPerNet * nNetsPerBatch);
	std::cout << "Max : " << maxS << " , avg : " << avgS << std::endl;
	rankArray(rawScores.data(), ranks, nNetsPerBatch);

	// Accumulate gradients so that the output of the generating meta-networks more closely ressembles
	// the elite networks in their respective neighborhoods. TODO globally instead ? In parallel of an
	// anti-contraction term ?
	for (int i = 0; i < (int) ((float)nNetsPerBatch * elitePercentage); i++) {
		examplifyNet(nets[ranks[i]]);
	}
	
	/*
	torch::optim::OptimizerOptions options;
	options.set_lr(lr);
	optimizer->param_groups()[0].set_options(std::make_unique<torch::optim::OptimizerOptions>(options));
	*/ // TODO fishy

	optimizer->step();

	delete[] nets;
}


// tensor.index, tensor.cat, tensor.slice and tensor.repeat let gradients flow through 
// (if the initial tensor had requires_grad set to true).
void Generator::accumulateGrads(int layer, float* Xs, float* X0) {

	torch::Tensor matTargets[N_MATRICES];
	// +1 for the compilation not to throw an error if N_ARRAYS=0. last slot is unused
	torch::Tensor arrTargets[N_ARRAYS + 1]; 

	// fill matTargets and arrTargets
	{
		torch::NoGradGuard no_grad;

		torch::Tensor originalInput = torch::from_blob(X0, { 1, gaussianVecSize });

		// colEmb0, colEmb1, ... colEmb(nCols-1), lineEmb0, ... lineEmb(nLines-1).
		torch::Tensor linesAndColsEmbeddings = matrixators[layer].forward(originalInput);

		// MATRICES
		{
			torch::Tensor colEmbeddings1D = linesAndColsEmbeddings.index({ 0, torch::indexing::Slice(0, colEmbS[layer] * nCols[layer]) });
			torch::Tensor colEmbeddings = colEmbeddings1D.repeat({ nLines[layer], 1 });

			torch::Tensor lineEmbeddings = torch::zeros({ nLines[layer], lineEmbS[layer] }, torch::dtype(torch::kFloat32));
			for (int i = 0; i < nLines[layer]; i++) {
				int i0 = colEmbS[layer] * nCols[layer] + i * lineEmbS[layer];
				for (int j = 0; j < lineEmbS[layer]; j++) {
					lineEmbeddings.index_put_({ i,j }, linesAndColsEmbeddings.index({ 0, i0 + j }));
				}
			}
			torch::Tensor matSpecialistInput = torch::cat({ colEmbeddings, lineEmbeddings }, 1);

			for (int matSpeID = 0; matSpeID < N_MATRICES; matSpeID++)
			{
				matTargets[matSpeID] = matSpecialists[layer * N_MATRICES + matSpeID].forward(matSpecialistInput).repeat({ nUpdatedPoints, 1 });
			}
		}

		if (N_ARRAYS > 0)
		{
			for (int arrSpeID = 0; arrSpeID < N_ARRAYS; arrSpeID++)
			{
				arrTargets[arrSpeID] = arrSpecialists[layer * N_ARRAYS + arrSpeID].forward(linesAndColsEmbeddings).repeat({ nUpdatedPoints, 1 });
			}
		}
	}


	// forward pass with gradients on the set of modified seeds.
	{

		torch::Tensor newInputs = torch::from_blob(X0, { nUpdatedPoints , gaussianVecSize }, 
			torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)
		);

		int intendedOutputsID = 0;

		// colEmb0, colEmb1, ... colEmb(nCols-1), lineEmb0, ... lineEmb(nLines-1).
		torch::Tensor linesAndColsEmbeddings = matrixators[layer].forward(newInputs);

		// MATRICES : TODO check the grads.
		{
			int start = colEmbS[layer] * nCols[layer];
			int end = start + lineEmbS[layer] * nLines[layer];
			
			// .requires_grad(true) not needed because it becomes true after linesAndColsEmbeddings's
			// subparts have been sliced into it. It even causes an error.
			torch::Tensor matSpecialistInput = torch::zeros(
				{ nLines[layer] * nUpdatedPoints, start + lineEmbS[layer] }, 
				torch::TensorOptions().dtype(torch::kFloat32) 
			);
			
			matSpecialistInput.slice(1, 0, start) = linesAndColsEmbeddings.slice(1, 0, start).repeat_interleave(nLines[layer], 0);


			for (int s = 0; s < nUpdatedPoints; s++) 
			{
				int offset = nLines[layer] * s;
				matSpecialistInput.slice(0, offset, offset + nLines[layer]).slice(1, start, start+lineEmbS[layer]) =
					linesAndColsEmbeddings.select(0, s).slice(0, start, end).view({ nLines[layer], lineEmbS[layer] });

				/*for (int i = 0; i < nLines[layer]; i++) {
					int j0 = start + i * lineEmbS[layer];
					matSpecialistInput.select(0, offset + i).slice(0, start, end) =
						linesAndColsEmbeddings.select(0, s).slice(0, j0, j0+ lineEmbS[layer]);
					}
				}*/

			}

			/*
			torch::Tensor colEmbeddings = linesAndColsEmbeddings.slice(1, 0, start).repeat_interleave(nLines[layer], 0);

			// colEmbeddings now should have shape:
			// [ [colEmbsSeed1],
			//   [colEmbsSeed1],
			//   ...,  nLines times
			//   [colEmbsSeed2],
			//   [colEmbsSeed2],
			//   ...
			//   ...
			//   [colEmbsSeednUpdatedPoints] ]
			// i.e. nUpdatedPoints * nLines lines, colEmbS * nCols columns.

			torch::Tensor lineEmbeddings = torch::zeros({ nLines[layer] * nUpdatedPoints, lineEmbS[layer] }, torch::dtype(torch::kFloat32));
			for (int s = 0; s < nUpdatedPoints; s++) {
				int offset = nLines[layer] * s;
				for (int i = 0; i < nLines[layer]; i++) {
					int j0 = colEmbS[layer] * nCols[layer] + i * lineEmbS[layer];
					for (int j = 0; j < lineEmbS[layer]; j++) {
						lineEmbeddings.index_put_({ i+offset,j }, linesAndColsEmbeddings.index({ s, j0 + j }));
					}
				}
			}
			torch::Tensor matSpecialistInput = torch::cat({ colEmbeddings, lineEmbeddings }, 1);

			// matSpecialistInput now should have shape:
			// [ [colEmbsSeed1, lineEmb1Seed1],
			//   [colEmbsSeed1, lineEmb2Seed1],
			//   ...,  
			//   [colEmbsSeed1, lineEmbnLinesSeed1],
			//   [colEmbsSeed2, lineEmb1Seed2],
			//   [colEmbsSeed2, lineEmb2Seed2],
			//   ...
			//   ...
			//   [colEmbsSeednUpdatedPoints,  lineEmbnLinesSeednUpdatedPoints] ]
			// i.e. nUpdatedPoints * nLines rows, colEmbS * nCols + lineEmbS columns.
			*/

			for (int matSpeID = 0; matSpeID < N_MATRICES; matSpeID++)
			{
				torch::Tensor matSpecialistOutput = matSpecialists[layer*N_MATRICES+matSpeID].forward(matSpecialistInput);
				torch::Tensor out = torch::mse_loss(matSpecialistOutput, matTargets[matSpeID]);
				out.backward();
				 
				//out.backward({}, true);
				//out.backward({}, true, false);
				//out.backward({}, true, true);
				// 
				//out.backward({}, false, true);
			}

		}


		// ARRAYS :
		if (N_ARRAYS > 0) {
			for (int arrSpeID = 0; arrSpeID < N_ARRAYS; arrSpeID++)
			{
				torch::Tensor arrSpecialistOutput = arrSpecialists[layer * N_ARRAYS + arrSpeID].forward(linesAndColsEmbeddings);
				torch::Tensor out = torch::mse_loss(arrSpecialistOutput, arrTargets[arrSpeID]);
				out.backward({}, true, true);
			}

		}
	}
}


void Generator::examplifyNet(Network* net)
{
	float* seeds = new float[nUpdatedPoints*gaussianVecSize];

	int nId = 0;
	for (int l = 0; l < nLayers; l++) {
		for (int n = 0; n < nNodesAtLayer[l]; n++) {

			for (int i = 0; i < nUpdatedPoints; i++)
			{
				for (int j = 0; j < gaussianVecSize; j++) {
					seeds[i* gaussianVecSize + j] = net->seed[nId * gaussianVecSize + j] + NORMAL_01 * updateRadius;
				}
			}
			accumulateGrads(l, seeds, &net->seed[nId]);
			nId++;
		}
	}

	delete[] seeds;
}

void Generator::flattenNetwork(Node** nodes, Network* n) 
{
	Node** prev = new Node * [nNodesAtLayer[nLayers - 1]];
	Node** curr = new Node * [nNodesAtLayer[nLayers - 1]];

	prev[0] = n->topNode.get();
	nodes[0] = n->topNode.get();
	int nId = 1;

	for (int i = 0; i < nLayers; i++) 
	{
		int nc = nChildrenAtLayer[i];
		for (int j = 0; j < nNodesAtLayer[i]; j++)
		{
			for (int k = 0; k < nc; k++) 
			{
				nodes[nId] = (curr[j * nc + k] = &prev[j]->children[k]);
				nId++;
			}
		}
		Node** temp = prev;
		prev = curr;
		curr = temp;
	}

	delete[] prev;
	delete[] curr;
}

void Generator::fillNode(Node* node, int layer, float* seed) 
{
	torch::NoGradGuard no_grad;

	torch::Tensor input = torch::from_blob(seed, { 1, gaussianVecSize });

	// colEmb0, colEmb1, ... colEmb(nCols-1), lineEmb0, ... lineEmb(nLines-1).
	torch::Tensor linesAndColsEmbeddings = matrixators[layer].forward(input);

	// MATRICES :
	{
		torch::Tensor colEmbeddings1D = linesAndColsEmbeddings.index({ 0, torch::indexing::Slice(0, colEmbS[layer] * nCols[layer]) });
		torch::Tensor colEmbeddings = colEmbeddings1D.repeat({ nLines[layer], 1 });

		torch::Tensor lineEmbeddings = torch::zeros({ nLines[layer], lineEmbS[layer] }, torch::dtype(torch::kFloat32));
		for (int i = 0; i < nLines[layer]; i++) {
			int i0 = colEmbS[layer] * nCols[layer] + i * lineEmbS[layer];
			for (int j = 0; j < lineEmbS[layer]; j++) {
				lineEmbeddings.index_put_({ i,j }, linesAndColsEmbeddings.index({ 0, i0 + j }));
			}
		}
		torch::Tensor matSpecialistInput = torch::cat({ colEmbeddings, lineEmbeddings }, 1);


		// This block is horrendous but I dont see any way around it without rearchitecturing
		auto fillMat = [this, &matSpecialistInput, layer, node](int speID)
		{

			torch::Tensor matSpecialistOutput = matSpecialists[layer * N_MATRICES + speID].forward(matSpecialistInput);
			
			auto accessor = matSpecialistOutput.accessor<float, 2>();

			auto fillLines = [&accessor, this, speID, layer](int line0, InternalConnexion* co) {
				float* mat = nullptr;
				bool is01;

				// Switch does not accomodate conditional compilation easily.
				{
					int i = 0;
					if (i++ == speID) {
						mat = co->A.get(); is01 = false;
					}
					else if (i++ == speID) {
						mat = co->B.get(); is01 = false;
					}
					else if (i++ == speID) {
						mat = co->C.get(); is01 = false;
					}
					else if (i++ == speID) {
						mat = co->D.get(); is01 = false;
					}
					else if (i++ == speID) {
						mat = co->eta.get();  is01 = true;
					}
					else if (i++ == speID) {
						mat = co->alpha.get(); is01 = false;
					}
					else if (i++ == speID) {
						mat = co->gamma.get();  is01 = true;
					}

#ifndef RANDOM_WB
					else if (i++ == speID) {
						mat = co->w.get(); is01 = false;
					}
#endif 

#ifdef OJA
					else if (i++ == speID) {
						mat = co->delta.get();  is01 = true;
					}
#endif
				}

				float* debug = &accessor[0][0];
				int matID = 0;
				if (is01) {
					for (int i = line0; i < line0 + co->nLines; i++) {
						for (int j = 0; j < nCols[layer]; j++) {
							float invTau = powf(2.0f, -(5.0f * accessor[i][j] + 2.0f));
							mat[matID] = 1.0f - powf(2.0f, -invTau);
							matID++;
						}
					}
				}
				else {
					for (int i = line0; i < line0 + co->nLines; i++) {
						for (int j = 0; j < nCols[layer]; j++) {
							mat[matID] = 2.0f * accessor[i][j];
							matID++;
						}
					}
				}
			};

			int l0 = 0;
			fillLines(l0, &node->toOutput);
			l0 += node->outputSize;
			fillLines(l0, &node->toModulation);
			l0 += MODULATION_VECTOR_SIZE;
			fillLines(l0, &node->toChildren);
		};

		int matSpeID = 0;
		fillMat(matSpeID++); // A
		fillMat(matSpeID++); // B
		fillMat(matSpeID++); // C
		fillMat(matSpeID++); // D
		fillMat(matSpeID++); // eta
		fillMat(matSpeID++); // alpha
		fillMat(matSpeID++); // gamma
#ifndef RANDOM_WB
		fillMat(matSpeID++); // w
#endif 
#ifdef OJA
		fillMat(matSpeID++); // delta	
#endif 
	}


	// ARRAYS :
	if (N_ARRAYS > 0)
	{

		auto fillArr = [this, &linesAndColsEmbeddings, layer, node](int speID)
		{

			torch::Tensor arrSpecialistOutput = arrSpecialists[layer * N_ARRAYS + speID].forward(linesAndColsEmbeddings);

			auto accessor = arrSpecialistOutput.accessor<float, 2>();

			auto fillSubArr = [&accessor, this, speID](int i0, InternalConnexion* co) {
				float* mat = nullptr;
				bool is01 = false;


				int i = 0;

#ifndef RANDOM_WB
				if (i++ == speID) {
					mat = co->biases.get(); is01 = false;
				}
#endif

#ifdef STDP
				if (i++ == speID) {
					mat = co->STDP_mu.get();  is01 = true;
				}
				else if (i++ == speID) {
					mat = co->STDP_lambda.get();  is01 = true;
				}
#endif 
				if (is01) {
					for (int j = i0; j < i0 + co->nLines; j++) {
						float invTau = powf(2.0f, -(5.0f * accessor[0][j] + 2.0f));
						mat[j - i0] = 1.0f - powf(2.0f, -invTau);
					}
				}
				else {
					for (int j = i0; j < i0 + co->nLines; j++) {
						mat[j - i0] = 3.0f * accessor[0][j];
					}
				}

			};

			int i0 = 0;
			fillSubArr(i0, &node->toOutput);
			i0 += node->outputSize;
			fillSubArr(i0, &node->toModulation);
			i0 += MODULATION_VECTOR_SIZE;
			fillSubArr(i0, &node->toChildren);
		};

		int arrSpeID = 0;

#ifndef RANDOM_WB
		fillArr(arrSpeID++); // biases
#endif

#ifdef STDP
		fillArr(arrSpeID++); // mu
		fillArr(arrSpeID++); // lambda
#endif 

	}
}

Network* Generator::createNet()
{

	float* seed = new float[seedSize];
	for (int i = 0; i < seedSize; i++) {
		seed[i] = NORMAL_01;
	}

	Network* net = new Network(netInSize, netOutSize, seed);
	net->createPhenotype(
		nodesInSizeAtLayer.get(),
		nodesOutSizeAtLayer.get(),
		nChildrenAtLayer.get(),
		nNodesAtLayer.get(),
		nLayers
	);

	Node** nodes = new Node*[seedSize / gaussianVecSize]; // seedSize / gaussianVecSize = total n Nodes

	flattenNetwork(nodes, net);
	
	int nId = 0;
	for (int l = 0; l < nLayers; l++) {
		for (int n = 0; n < nNodesAtLayer[l]; n++) {
			fillNode(nodes[nId], l, seed);
			seed += gaussianVecSize;
			nId++;
		}
	}

	delete[] nodes;

	return net;
}

void Generator::save() 
{

}

void Generator::save1Net() 
{

}
