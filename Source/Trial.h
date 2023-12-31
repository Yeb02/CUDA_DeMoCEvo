#pragma once

#include <vector>
#include <iostream>

#include "VirtualTrial.h"

#include "DeMoCEvoCore.h"


/* v1 and v2 are binary vectors (-1 or 1), randomly initialized. The trial is split in 3 phases. In the first one,
the observation is the vector v1. In the second, it is v2. In the third, the observation is a vector of 0s.
During the last phase, the expected output of the network is the termwise XOR of v1 and v2. Only the sign of
the network's output is used. There are 2^(2*vSize) different trials possible. 
Note that be it with or without CONTINUOUS_LEARNING, lifelong learning on this task is completely useless.
Therefore, it can be seen as a robustness test, a benchmark on catastrophic forgetting (even though
what is to be "remembered" was not learned during the lifetime, but a genetic trait).
*/
class XorTrial : public Trial {

public:
	// Required network sizes: INPUT_NODE = vectorSize, output = vectorSize.
	XorTrial(int vectorSize, int delay);
	void step(const float* actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;
	void outerLoopUpdate(void* data) override {};

private:
	int vSize;
	int delay;
	std::vector<bool> v1, v2, v1_xor_v2;
};


// Classic CartPole, adapted from the python version of 
// https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
class CartPoleTrial : public Trial {

public:
	CartPoleTrial(bool continuousControl);
	void step(const float* actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;
	void outerLoopUpdate(void* data) override {};
	
	// or 30000... Gym's baseline is either 200 or 500, which is quite short with tau=0.02.
	static const int STEP_LIMIT = 1000; 

private:
	bool continuousControl;
	float x, xDot, theta, thetaDot;
	float x0, xDot0, theta0, thetaDot0;

	void setObservations();
};



// as per Soltoggio et al. (2008)
class TMazeTrial : public Trial{
public:
	TMazeTrial(bool switchesSide);
	void step(const float* actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;
	void outerLoopUpdate(void* data) override {
		switchesSide = *static_cast<bool*>(data);
	};

	static const int corridorLength = 5;
	static const int nInferencesBetweenEnvSteps = 3;

private:
	// the network performs nInferencesBetweenEnvSteps inferences between each step of the environment,
	// to give enough time for information to flow through.
	int inferenceStep;

	// to be set in main.cpp's loop at each step.
	bool switchesSide;

	void subTrialReset();

	bool wentLeft;
	int nSubTrials;
};


// Observation are: [cartX, cosTheta1 ,sinTheta1, cosTheta2, ... sinThetaNLinks)], where cosines and sines  
// are with respect to the global axis. The zero radians is the trigonometric standard, horizontal right.
// The network is not fed the speed of the arms. It model will have to infer it by itself. 
class NLinksPendulumTrial : public Trial {

public:
	NLinksPendulumTrial(bool continuousControl, int nJoins);
	void step(const float* actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;
	void outerLoopUpdate(void* data) override {};

	static const int STEP_LIMIT = 1000;

private:
	bool continuousControl;
	int nLinks;

	// initial values
	float x0;
	std::unique_ptr<float[]> thetas0;

	// Redundant but useful
	std::unique_ptr<float[]> thetas;

	// Cartesian state
	std::unique_ptr<float[]> xs, vxs, ys, vys;

	// Positions at previous step.
	std::unique_ptr<float[]> pxs, pys;
};


// The trial is split in two phases. In the first one, the network is presented with a set of 
// motif-response pairs. In the second phase, the network is presented with motifs from the first
// step, but the responses are set to 0. The task of the network is to output the associated response 
// for each motif. The parameter binary of the constructor determines whether the motif-response pair 
// should be binary {-1,1}, or continuous [-1,1]. 
class MemoryTrial : public Trial {

public:
	MemoryTrial(int nMotifs, int motifSize, int responseSize, bool binary=true);
	void step(const float* actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;
	void outerLoopUpdate(void* data) override {};


private:
	int nMotifs, motifSize, responseSize;

	bool binary;

	// matrix of size nMotifs * (motifSize+responseSize)
	std::unique_ptr<float[]> motifResponsePairs;
};