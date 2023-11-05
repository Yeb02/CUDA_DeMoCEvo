#pragma once

#include "System.h"



System::System(SystemParameters& params, Trial* _trial, bool libtorchUsesCuda):
	trial(_trial)
{ 
	setParameters(params);

	ConnexionGenerator::nPerturbations = nPerturbations;

	torch::Device* device;
	if (torch::cuda::is_available() && libtorchUsesCuda) {
		std::cout << "Libtorch uses CUDA" << std::endl;
		device = new torch::Device(torch::kCUDA);
	}
	else {
		std::cout << "Libtorch does not use CUDA" << std::endl;
		device = new torch::Device(torch::kCPU);
	}
	
	rootGeneratorNode = std::make_unique<GeneratorNode>(inSizePerL, outSizePerL, nChildrenPerL, seedSize, learningRate, device);


	coefficients = new float[nPerturbations];
	teachers.resize(0);


	agentsFitnesses = new float[2 * nPerturbations + 1];
	agentsScores = new float* [nEvaluationTrials+1]; // +1 for temp calculations
	for (int i = 0; i < nEvaluationTrials+1; i++) agentsScores[i] = new float[2 * nPerturbations + 1];

	teachersFitnesses = new float[nTeachers];
	teachersScores = new float* [nSupervisedTrials+1]; // +1 for temp calculations
	for (int i = 0; i < nSupervisedTrials+1; i++) teachersScores[i] = new float[nTeachers];



#ifdef ONE_MATRIX
	int activationArraySize = 0;
#else
	int activationArraySize = outSizePerL[0];
#endif
	int activationArraySize = outSizePerL[0];
	std::vector<int> nModulesPerNetworkLayer(treeDepth);
	nModulesPerNetworkLayer[0] = 1;
	for (int l = 0; l < treeDepth; l++)
	{
		int nc = nChildrenPerL[l];
		if (l < treeDepth - 1) nModulesPerNetworkLayer[l + 1] = nc * nModulesPerNetworkLayer[l];

		int cOs = nc == 0 ? 0 : outSizePerL[l + 1];


#ifdef ONE_MATRIX
		int cIs = nc == 0 ? 0 : inSizePerL[l + 1];
		activationArraySize += nModulesPerNetworkLayer[l] * (inSizePerL[l] + outSizePerL[l] + (cOs + cIs) * nc);
#else
		activationArraySize += nModulesPerNetworkLayer[l] * (inSizePerL[l] + cOs * nc);
#endif
	}


	Network::activationArraySize = activationArraySize;
	Network::inS = inSizePerL;
	Network::outS = outSizePerL;
	Network::nC = nChildrenPerL;
	Network::nLayers = treeDepth;

	agents = new Network * [2 * nPerturbations + 1];
	for (int i = 0; i < (2 * nPerturbations + 1); i++) {
		agents[i] = new Network(rootGeneratorNode.get());
	}
}


void System::teachAndEvaluate() 
{
	for (int i = 0; i < 2 * nPerturbations + 1; i++) {
		int pID = i == 2 * nPerturbations ? -1 : i / 2;
		bool negative = ((i % 2) == 1); // negative if i is odd

		
		agents[i]->generatePhenotype(rootGeneratorNode.get(), pID, negative);

		for (int j = 0; j < nSupervisedTrials; j++) {
			Network teacher(*teachers[j % (int)teachers.size()]); TODO when does the teacher learn;

			float* teacherOutput = teacher.getOutput();

			teacher.preTrialReset();
			agents[i]->preTrialReset();
			
			trial->reset(false);

			while (!trial->isTrialOver)
			{
				teacher.step(trial->observations.data(), false);
				agents[i]->step(trial->observations.data(), true, teacherOutput);
				trial->step(teacherOutput);
			}

			teachersScores[j][j % (int)teachers.size()] = trial->score;
		}

		for (int j = 0; j < nEvaluationTrials; j++) {
			agents[i]->preTrialReset();

			trial->reset(false);

			while (!trial->isTrialOver)
			{
				agents[i]->step(trial->observations.data(), false);
				trial->step(agents[i]->getOutput());
			}

			agentsScores[j][i] = trial->score;
		}
	}
}


void System::computeCoefficients() 
{
	int size = 2 * nPerturbations + 1;

	std::fill(agentsFitnesses, agentsFitnesses + size, 0.0f);

	for (int i = 0; i < nEvaluationTrials; i++) {
		rankArray(agentsScores[i], agentsScores[nEvaluationTrials], size);
		for (int j = 0; j < size; j++) {
			agentsFitnesses[j] += agentsScores[nEvaluationTrials][j];
		}
	}

	// also sets bestAgentID.
	rankArray(agentsFitnesses, agentsFitnesses, size, &bestAgentID);

	for (int i = 0; i < nPerturbations; i++) 
	{
		float d = .5f * (agentsFitnesses[2 * i] + agentsFitnesses[2 * i + 1]) - agentsFitnesses[size - 1];
		float f = powf(potentialChange, d) / (float) nPerturbations;
		coefficients[i] = (agentsFitnesses[2 * i] - agentsFitnesses[2 * i + 1]) * f;
	}
}


void System::updateTeachers() 
{
	int size = teachers.size();

	if (size < nTeachers) {
		teachers.emplace_back(*agents[bestAgentID]);
		return;
	}


	std::fill(teachersFitnesses, teachersFitnesses + size, 0.0f);

	for (int i = 0; i < nSupervisedTrials; i++) {
		rankArray(teachersScores[i], teachersScores[nSupervisedTrials], size);
		for (int j = 0; j < size; j++) {
			teachersFitnesses[j] += teachersScores[nSupervisedTrials][j];
		}
	}


	std::vector<int> positions(size);
	for (int i = 0; i < size; i++) {
		positions[i] = i;
	}

	// sort position by ascending value.
	float* src = teachersFitnesses;
	std::sort(positions.begin(), positions.end(), [src](int a, int b) -> bool
		{
			return src[a] < src[b];
		}
	);


	teachers[positions[0]]->deepCopy(*agents[bestAgentID]);
}


void System::evolve(int nSteps)
{
	for (int s = 0; s < nSteps; s++)
	{

		rootGeneratorNode->zeroGrad();

		for (int i = 0; i < nBatches; i++)
		{
			rootGeneratorNode->generatePerturbations(perturbationMagnitude);

			teachAndEvaluate();
			
			// Happens before updateTeachers() so that the best agent at this step, that will replace the worst 
			// teacher, is known.
			computeCoefficients();

			updateTeachers();

			if (i == 0) log(s);

			rootGeneratorNode->accumulateGradient(coefficients);
		}

		rootGeneratorNode->optimizerStep();
	}
}


