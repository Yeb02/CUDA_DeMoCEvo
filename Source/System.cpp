#pragma once

#include "System.h"



System::System(SystemParameters& params, Trial* _trial):
	trial(_trial)
{ 
	setParameters(params);

	ConnexionGenerator::nPerturbations = nPerturbations;

	rootGeneratorNode = std::make_unique<GeneratorNode>(inSizePerL, outSizePerL, nChildrenPerL, seedSize, learningRate);
	
	coefficients = new float[nPerturbations];
	teachers.resize(0);


	int activationArraySize = outSizePerL[0];
	std::vector<int> nModulesPerNetworkLayer(treeDepth);
	nModulesPerNetworkLayer[0] = 1;
	for (int l = 0; l < treeDepth; l++)
	{
		int nc = nChildrenPerL[l];
		if (l < treeDepth - 1) nModulesPerNetworkLayer[l + 1] = nc * nModulesPerNetworkLayer[l];

		int cOs = nc == 0 ? 0 : outSizePerL[l + 1];

		activationArraySize += nModulesPerNetworkLayer[l] * (inSizePerL[l] + cOs * nc);
	}


	Network::activationArraySize = activationArraySize;
	Network::inS = inSizePerL;
	Network::outS = outSizePerL;
	Network::nC = nChildrenPerL;
	Network::nLayers = treeDepth;
}


void System::teachAndEvaluate() 
{
	for (int i = 0; i < 2 * nPerturbations + 1; i++) {
		Network agent;

		int pID = i == 2 * nPerturbations ? -1 : i / 2;
		bool negative = i % 2;

		agent.createPhenotype(rootGeneratorNode.get(), pID, negative);

		for (int j = 0; j < nSupervisedTrials; j++) {
			Network teacher(*teachers[j % (int)teachers.size()]);

			float* teacherOutput = teacher.getOutput();

			teacher.preTrialReset();
			agent.preTrialReset();
			
			trial->reset(false);

			while (!trial->isTrialOver)
			{
				teacher.step(trial->observations.data(), false);
				agent.step(trial->observations.data(), true, teacherOutput);
				trial->step(teacherOutput);
			}
		}

		for (int j = 0; j < nEvaluationTrials; j++) {
			agent.preTrialReset();

			trial->reset(false);

			while (!trial->isTrialOver)
			{
				agent.step(trial->observations.data(), false);
				trial->step(agent.getOutput());
			}

			agentsScores[j][i] = trial->score;
		}
	}
}


void System::computeCoefficients() {

}


void System::updateTeachers() {


}


void System::evolve(int nSteps)
{
	for (int s = 0; s < nSteps; s++)
	{

		rootGeneratorNode->zeroGrad();

		for (int i = 0; i < nBatches; i++)
		{

			teachAndEvaluate();
			
			updateTeachers();

			// must be called after updateTeachers, because it modifies the agentScores array and updateTeachers needs it raw.
			computeCoefficients();

			if (i == 0) log(s);

			rootGeneratorNode->accumulateGrad(coefficients);
		}

		rootGeneratorNode->optimizerStep();
	}
}


