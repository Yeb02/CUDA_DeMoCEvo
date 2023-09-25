#pragma once

#include "Node.h"


// Should never be called.
Node::Node() 
{ //TODO donner un constructeur vide, puisqu il ne sera jamais appelé
	__debugbreak();

	inputSize = 0;
	outputSize = 0;

	inputArray = nullptr;
	destinationArray = nullptr;
	
#ifdef STDP
	accumulatedPreSynActs = nullptr;
#endif

#ifdef SATURATION_PENALIZING
	averageActivation = nullptr;
	globalSaturationAccumulator = nullptr;
#endif
};

Node::Node(int* inS, int* outS, int* nC) :
	inputSize(inS[0]), outputSize(outS[0]),
	toModulation(MODULATION_VECTOR_SIZE, computeNCols(inS, outS, nC[0])),
	toChildren(nC[0] > 0 ? nC[0] * inS[1] : 0, computeNCols(inS, outS, nC[0])),
	toOutput(outS[0], computeNCols(inS, outS, nC[0]))
{

	children.reserve(nC[0]);
	for (int i = 0; i < nC[0]; i++) {
		children.emplace_back(inS + 1, outS + 1, nC + 1);
	}

	inputArray = nullptr;
	destinationArray = nullptr;

#ifdef STDP
	accumulatedPreSynActs = nullptr;
#endif

#ifdef SATURATION_PENALIZING
	averageActivation = nullptr;
	globalSaturationAccumulator = nullptr;
#endif
}


void Node::setArrayPointers(float** post_syn_acts, float** pre_syn_acts, float** aa, float** acc_pre_syn_acts) {

	// TODO ? if the program runs out of heap memory, one could make it so that a node does not store its own 
	// output. But prevents in place matmul, and complexifies things.

	inputArray = *post_syn_acts;
	destinationArray = *pre_syn_acts;


	*post_syn_acts += inputSize + MODULATION_VECTOR_SIZE;
	*pre_syn_acts += outputSize + MODULATION_VECTOR_SIZE;


#ifdef SATURATION_PENALIZING
	averageActivation = *aa;
	*aa += MODULATION_VECTOR_SIZE;
#endif

#ifdef STDP
	accumulatedPreSynActs = *acc_pre_syn_acts;
	*acc_pre_syn_acts += outputSize + MODULATION_VECTOR_SIZE;
#endif

	for (int i = 0; i < children.size(); i++) {
		*post_syn_acts += children[i].outputSize;
		*pre_syn_acts += children[i].inputSize;
#ifdef STDP
		* acc_pre_syn_acts += children[i].inputSize;
#endif
#ifdef SATURATION_PENALIZING
		* aa += children[i].inputSize;
#endif
	}


	for (int i = 0; i < children.size(); i++) {
		children[i].setArrayPointers(post_syn_acts, pre_syn_acts, aa, acc_pre_syn_acts);
	}
}


void Node::preTrialReset() {

	for (int i = 0; i < children.size(); i++) {
		children[i].preTrialReset();
	}

	toChildren.zeroEH();
	toModulation.zeroEH();
	toOutput.zeroEH();


#ifdef RANDOM_WB
	toChildren.randomInitWB();
	toModulation.randomInitWB();
	toOutput.randomInitWB();
#endif

#if defined(ZERO_WL_BEFORE_TRIAL)
	toChildren.zeroWlifetime();
	toModulation.zeroWlifetime();
	toOutput.zeroWlifetime();
#endif 

}


#ifdef SATURATION_PENALIZING
void Node::setglobalSaturationAccumulator(float* globalSaturationAccumulator) {
	this->globalSaturationAccumulator = globalSaturationAccumulator;
	for (int i = 0; i < children.size(); i++) {
		children[i].setglobalSaturationAccumulator(globalSaturationAccumulator);
	}
}
#endif


void Node::forward() {

#ifdef SATURATION_PENALIZING
	constexpr float saturationExponent = 6.0f;
#endif



#ifdef DROPOUT
	toChildren.dropout();
	toOutput.dropout();
	toModulation.dropout();
#endif

	// STEP 1 to 4: propagate and call children's forward.
	// 1_Modulation -> 2_Complex -> 3_Modulation -> 4_output
	// This could be done simultaneously for all types, but doing it this way drastically speeds up information transmission
	// through the network. 


	// These 3 lambdas, hopefully inline, avoid repetition, as they are used for each child type.

	auto propagate = [this](InternalConnexion* co, float* inputArray)
	{
		int nl = co->nLines;
		int nc = co->nColumns;
		int matID = 0;

		float* H = co->H.get();
		float* wLifetime = co->wLifetime.get();
		float* alpha = co->alpha.get();

		float* w = co->w.get();
		float* b = co->biases.get();


		for (int i = 0; i < nl; i++) {
			inputArray[i] = b[i];
			for (int j = 0; j < nc; j++) {
				// += (H * alpha + w + wL) * prevAct
				inputArray[i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * inputArray[j];
				matID++;
			}
		}


	};

	auto hebbianUpdate = [this](InternalConnexion* co, float* inputArray) {
		int nl = co->nLines;
		int nc = co->nColumns;
		int matID = 0;

		float* A = co->A.get();
		float* B = co->B.get();
		float* C = co->C.get();
		float* D = co->D.get();
		float* eta = co->eta.get();
		float* H = co->H.get();
		float* E = co->E.get();

		float* wLifetime = co->wLifetime.get();
		float* gamma = co->gamma.get();
		float* alpha = co->alpha.get();


#ifdef OJA
		float* delta = co->delta.get();
		float* w = co->w.get();
#endif


		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {
				wLifetime[matID] = (1 - gamma[matID]) * wLifetime[matID] + gamma[matID] * H[matID] * alpha[matID] * totalM[1]; // TODO remove ?

				E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
					(A[matID] * inputArray[i] * inputArray[j] + B[matID] * inputArray[i] + C[matID] * inputArray[j] + D[matID]);

#ifdef OJA
				E[matID] -= eta[matID] * inputArray[i] * inputArray[i] * delta[matID] * (w[matID] + alpha[matID] * H[matID] + wLifetime[matID]);
#endif

				H[matID] += E[matID] * totalM[0];
				H[matID] = std::clamp(H[matID], -1.0f, 1.0f);

				matID++;

			}
		}
	};

	auto applyNonLinearities = [](float* src, float* dst, int size
#ifdef STDP
		, float* acc_src, float* mu, float* lambda
#endif
		)
	{
#ifdef STDP
		for (int i = 0; i < size; i++) {
			acc_src[i] = acc_src[i] * (1.0f - mu[i]) + src[i]; // * mu[i] ? TODO
		}

		src = acc_src;
#endif

		for (int i = 0; i < size; i++) {
			dst[i] = tanhf(src[i]);

			if (src[i] != src[i] || dst[i] != dst[i]) {
				__debugbreak();
			}
		}

#ifdef STDP
		for (int i = 0; i < size; i++) {
			// acc_src's magnitude decreases when there is a significant activation.
			acc_src[i] -= lambda[i] * powf(dst[i], 2.0f * 1.0f + 1.0f); 
		}
#endif
	};



	// STEP 1: MODULATION  A
	{
		propagate(&toModulation, destinationArray + outputSize);
		applyNonLinearities(
			destinationArray + outputSize,
			inputArray + inputSize,
			MODULATION_VECTOR_SIZE
#ifdef STDP
			, accumulatedPreSynActs + outputSize, toModulation.STDP_mu.get(), toModulation.STDP_lambda.get()
#endif
		);
		hebbianUpdate(&toModulation, inputArray + inputSize);

		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			totalM[i] += inputArray[i + inputSize];
		}

#ifdef SATURATION_PENALIZING 
		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			float v = inputArray[i + inputSize];
			*globalSaturationAccumulator += powf(abs(v), saturationExponent);
			averageActivation[i] += v;
		}
#endif
	}


	// STEP 2: COMPLEX
	if (children.size() != 0) {
		float* ptrToInputs = destinationArray + outputSize + MODULATION_VECTOR_SIZE;
#ifdef STDP
		float* ptrToAccInputs = accumulatedPreSynActs + outputSize + MODULATION_VECTOR_SIZE;
#endif
		propagate(&toChildren, ptrToInputs);



		// Apply non-linearities
		int id = 0;
		for (int i = 0; i < children.size(); i++) {


			applyNonLinearities(
				ptrToInputs + id,
				children[i].inputArray,
				children[i].inputSize
#ifdef STDP
				, ptrToAccInputs + id, &toChildren.STDP_mu[id], &toChildren.STDP_lambda[id]
#endif
			);

#ifdef SATURATION_PENALIZING 
			// child post-syn input
			int i0 = MODULATION_VECTOR_SIZE + id;
			for (int j = 0; j < children[i].inputSize; j++) {
				float v = children[i].inputArray[j];
				*globalSaturationAccumulator += powf(abs(v), saturationExponent);
				averageActivation[i0 + j] += v;
			}
#endif

			id += children[i].inputSize;
		}

		// has to happen after non linearities but before forward, 
		// for children's output not to have changed yet.
		hebbianUpdate(&toChildren, ptrToInputs);


		// transmit modulation and apply forward, then retrieve the child's output.

		float* childOut = inputArray + inputSize + MODULATION_VECTOR_SIZE;
		for (int i = 0; i < children.size(); i++) {

			for (int j = 0; j < MODULATION_VECTOR_SIZE; j++) {
				children[i].totalM[j] = this->totalM[j];
			}

			children[i].forward();

			std::copy(children[i].destinationArray, children[i].destinationArray + children[i].outputSize, childOut);
			childOut += children[i].outputSize;
		}

	}


	// STEP 3: MODULATION B. 
	{
		propagate(&toModulation, destinationArray + outputSize);
		applyNonLinearities(
			destinationArray + outputSize,
			inputArray + inputSize,
			MODULATION_VECTOR_SIZE
#ifdef STDP
			, accumulatedPreSynActs + outputSize, toModulation.STDP_mu.get(), toModulation.STDP_lambda.get()
#endif
		);
		hebbianUpdate(&toModulation, inputArray + inputSize);

		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			totalM[i] += inputArray[i + inputSize];
		}

#ifdef SATURATION_PENALIZING 
		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			float v = inputArray[i + inputSize];
			*globalSaturationAccumulator += powf(abs(v), saturationExponent);
			averageActivation[i] += v;
		}
#endif
	}


	// STEP 4: OUTPUT
	{
		propagate(&toOutput, destinationArray);


		applyNonLinearities(
			destinationArray,
			destinationArray,
			outputSize
#ifdef STDP
			, accumulatedPreSynActs, toOutput.STDP_mu.get(), toOutput.STDP_lambda.get()
#endif
		);

		hebbianUpdate(&toOutput, destinationArray);
	}

}
