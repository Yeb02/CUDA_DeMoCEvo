#pragma once

#include "Module.h"



Module::Module(GeneratorNode& _generator) :
	inputSize(_generator.inputSize), outputSize(_generator.outputSize), nChildren(_generator.nChildren),
#ifdef ONE_MATRIX
	parameters(_generator.nRows, _generator.nColumns, _generator.device),
#else
	toChildren(),
	toOutput(_generator.outputSize, _generator.nColumns, _generator.device),
#endif
	device(_generator.device)
{

	children.reserve(nChildren);
	for (int j = 0; j < nChildren; j++) {
		children.emplace_back(_generator.children[j]);
	}

#ifndef ONE_MATRIX
	toChildren.reserve(nChildren);
	for (int j = 0; j < nChildren; j++) {
		children.emplace_back(_generator.children[j]);
		toChildren.emplace_back(_generator.children[j].inputSize, _generator.nColumns, device);
	}	
#endif
}


void Module::generatePhenotype(GeneratorNode& _generator, int perturbationID, bool negative)
{


#ifdef ONE_MATRIX
	_generator.parametersGenerator.createPhenotypeArrays(
		parameters.matrices,
		parameters.vectors,
		perturbationID,
		negative
	);

#else
	_generator.toOutputGenerator.createPhenotypeArrays(
		toOutput.matrices,
		toOutput.vectors,
		perturbationID,
		negative
	);

	for (int j = 0; j < nChildren; j++) {
		_generator.toChildrenGenerators[j].createPhenotypeArrays(
			toChildren[j].matrices,
			toChildren[j].vectors,
			perturbationID,
			negative
		);
	}
#endif

	for (int j = 0; j < nChildren; j++) {
		children[j].generatePhenotype(_generator.children[j], perturbationID, negative);
	}
}


Module::Module(const Module& n) :
	inputSize(n.inputSize), outputSize(n.outputSize), nChildren(n.nChildren),
#ifdef ONE_MATRIX
	parameters(n.parameters),
#else
	toChildren(),
	toOutput(n.toOutput),
#endif
	device(n.device)
{
	children.reserve(nChildren);

	for (int j = 0; j < nChildren; j++) {
		children.emplace_back(n.children[j]);
	}

#ifndef ONE_MATRIX
	toChildren.reserve(nChildren);

	for (int j = 0; j < nChildren; j++) {
		toChildren.emplace_back(n.toChildren[j]);
	}
#endif
}


void Module::deepCopy(const Module& n)
{

#ifdef ONE_MATRIX
	parameters.deepCopy(n.parameters);
#else
	toOutput.deepCopy(n.toOutput);
	for (int j = 0; j < nChildren; j++) {
		children[j].deepCopy(n.children[j]);
		toChildren[j].deepCopy(n.toChildren[j]);
	}
#endif

	for (int j = 0; j < nChildren; j++) {
		children[j].deepCopy(n.children[j]);
	}
}


void Module::setArrayPointers(float** ptr_activations, float** ptr_accumulators, float* outActivations, float* outAccumulators)
{

#ifdef ONE_MATRIX

	outCinActivations = torch::from_blob(*ptr_activations , { parameters.nRows,1 }, torch::TensorOptions().device(*device));
	outCinAccumulators = torch::from_blob(*ptr_accumulators, { parameters.nRows,1 }, torch::TensorOptions().device(*device));

	outputActivations = torch::from_blob(*ptr_activations, { outputSize,1 }, torch::TensorOptions().device(*device));
	outputAccumulators = torch::from_blob(*ptr_accumulators, { outputSize,1 }, torch::TensorOptions().device(*device));

	*ptr_activations += parameters.nRows;
	*ptr_accumulators += parameters.nRows;

	inCoutActivations = torch::from_blob(*ptr_activations, { parameters.nColumns,1 }, torch::TensorOptions().device(*device));
	inCoutAccumulators = torch::from_blob(*ptr_accumulators, { parameters.nColumns,1 }, torch::TensorOptions().device(*device));

	inputActivations = torch::from_blob(*ptr_activations, { inputSize,1 }, torch::TensorOptions().device(*device));
	inputAccumulators = torch::from_blob(*ptr_accumulators, { inputSize,1 }, torch::TensorOptions().device(*device));


	*ptr_activations += parameters.nColumns;
	*ptr_accumulators += parameters.nColumns;

	for (int i = 0; i < children.size(); i++) {
		children[i].setArrayPointers(ptr_activations, ptr_accumulators, nullptr, nullptr);
	}

#else


	
	inCoutActivations = torch::from_blob(*ptr_activations, { toOutput.nColumns,1 }, torch::TensorOptions().device(*device));
	inCoutAccumulators = torch::from_blob(*ptr_accumulators, { toOutput.nColumns,1 }, torch::TensorOptions().device(*device));

	inputActivations = torch::from_blob(*ptr_activations, { inputSize,1 }, torch::TensorOptions().device(*device));
	inputAccumulators = torch::from_blob(*ptr_accumulators, { inputSize,1 }, torch::TensorOptions().device(*device));

	outputActivations = torch::from_blob(outActivations, { outputSize,1 }, torch::TensorOptions().device(*device));
	outputAccumulators = torch::from_blob(outAccumulators, { outputSize,1 }, torch::TensorOptions().device(*device));



	float* actCoutPtr = *ptr_activations + inputSize;
	float* accCoutPtr = *ptr_accumulators + inputSize;

	int offset = (nChildren == 0 ? 0 : nChildren * children[0].outputSize) + inputSize;
	*ptr_activations += offset;
	*ptr_accumulators += offset;

	for (int i = 0; i < children.size(); i++) {
		children[i].setArrayPointers(ptr_activations, ptr_accumulators, actCoutPtr, accCoutPtr);
		actCoutPtr += children[i].outputSize;// same for all i.
		accCoutPtr += children[i].outputSize;
	}
#endif
}


void Module::xUpdate_simultaneous()
{
	// TODO evolve per module ? If you change it here, it must also be updated in PC_Network::step
	constexpr float xlr = .5f;

	// The update of an activation consists in substracting the gradient of the energy * learning rate. This (-)gradient is accumulated in the _Accumulators arrays,
	// and is the difference of 2 numbers: the epsilon and the eta.
	// 
	// epsilon_l = (X_l - bias_l+1 + theta_l+1*f(x_l+1))   and * x_lr_l 
	// eta_l     = (theta_l-1.transposed * epsilon_l-1) * f'(x_l)
	// -grad = eta - epsilon
	// x -= lr * grad
	//
	// "This" handles the update of its children's inputs and outputs. If this is the root node, the output and input of this are managed by the PC_Network.


#ifdef ONE_MATRIX
	// f(X_inCout) is recomputed each time it is needed because on GPU its is more costly than memory operations.

	// grad_outCin = - epsilon_outCin = (bias + theta*f(X_inCout) - X_outCin)     (* invSigmas_outCin #ifdef ACTIVATION_VARIANCE)
	outCinAccumulators = (  (parameters.vectors[0] + parameters.matrices[0].matmul(torch::tanh(inCoutActivations)) ) - outCinActivations
#ifdef ACTIVATION_VARIANCE
		) * parameters.vectors[1];
#else
		);
#endif


	// grad_inCout = eta_inCout = (thetaTransposed * epsilon_outCin) * f'(X_inCout)
	inCoutAccumulators = (parameters.matrices[0].transpose(0,1).matmul(outCinAccumulators))
		* (torch::square(torch::tanh(inCoutActivations)) - 1.0f); // outCinAccumulators holds -epsilon so f' is opposed.


	int start1 = inputSize;
	int start2 = outputSize;
	for (int i = 0; i < nChildren; i++)
	{

		// Transmit activations
		children[i].inputActivations = inCoutActivations.slice(0, start1, start1 + children[i].inputSize);
		children[i].outputActivations = outCinActivations.slice(0, start2, start2 + children[i].outputSize);

		// recursive call to the children's function.
		children[i].xUpdate_simultaneous();

		// Retrieve epsilon and eta
		inCoutAccumulators.slice(0, start1, start1 + children[i].inputSize) += children[i].inputAccumulators;
		outCinAccumulators.slice(0, start2, start2 + children[i].outputSize) += children[i].outputAccumulators;

		start1 += children[i].outputSize;
		start2 += children[i].inputSize;
	}

	// The actual activation's update
	inCoutActivations.slice(0, inputSize, start1) += xlr * inCoutAccumulators.slice(0, inputSize, start1);
	outCinAccumulators.slice(0, outputSize, start2) += xlr * outCinAccumulators.slice(0, outputSize, start2);

#else
	// When this function is called, inCoutAccumulators holds -epsilon_in in its first slots and outputAccumulators holds -eta_out.


	// grad_out += - epsilon_out = (bias + theta*f(X_inCout) - X_out)     (* invSigmas_out #ifdef ACTIVATION_VARIANCE)
	outputAccumulators += (((toOutput.vectors[0] + toOutput.matrices[0] * inCoutActivations.array().tanh().matrix()) - outputActivations
#ifdef ACTIVATION_VARIANCE
		).array() * toOutput.vectors[1].array()).matrix();
#else
		));
#endif

	// grad_inCout += eta_inCout = (thetaTransposed * epsilon_out) * f'(X_inCout)
	inCoutAccumulators -= ((toOutput.matrices[0].transpose() * outputAccumulators).array() * (1.0f - inCoutActivations.array().tanh().square())).matrix();

	for (int i = 0; i < nChildren; i++)
	{
		// grad_Cin = - epsilon_Cin = (bias + theta*f(X_inCout) - X_Cin)    (* invSigmas_cIn #ifdef ACTIVATION_VARIANCE)
		children[i].inputAccumulators.noalias() = (((toChildren[i].vectors[0] + toChildren[i].matrices[0] * inCoutActivations.array().tanh().matrix()) - children[i].inputActivations
#ifdef ACTIVATION_VARIANCE
			).array() * toChildren[i].vectors[1].array()).matrix();
#else
			));
#endif

		// grad_inCout += eta_inCout = (thetaTransposed * epsilon_Cin) * f'(X_inCout)
		inCoutAccumulators -= ((toChildren[i].matrices[0].transpose() * children[i].inputAccumulators).array() * (1.0f - inCoutActivations.array().tanh().square())).matrix();


		// recursive call to the children's functions.
		children[i].xUpdate_simultaneous();


		// The actual activation's update
		children[i].inputActivations += xlr * children[i].inputAccumulators;
		children[i].outputActivations += xlr * children[i].outputAccumulators;

	}


#endif

	return;
}


void Module::thetaUpdate_simultaneous()
{

	// the lr is injected at places where it minimizes the number of multiplications, therefore
	// the code does not exactly look like the equations.
	const float theta_b_lr = .0001f;
#ifdef ACTIVATION_VARIANCE
	const float sigma_lr = .0001f;
#endif


#ifdef ONE_MATRIX




	// epsilon_outCin = X_outCin - (bias + theta*f(X_inCout))     (* invSigmas_outCin #ifdef ACTIVATION_VARIANCE)
	outCinAccumulators = (outCinActivations - (parameters.vectors[0] + parameters.matrices[0].matmul(torch::tanh(inCoutActivations)))
#ifdef ACTIVATION_VARIANCE
		) * parameters.vectors[1];
#else
		);
#endif

#ifdef ACTIVATION_VARIANCE
	// Reminder that vectors[1] contains the inverses of the variances.
	// sigma += (invSigmas * (  X_outCin - (theta*f(X_inCout)+bias)  )^2  - 1) * sigma_lr = (sigmas * epsilon_outCin^2  - 1) * sigma_lr
	parameters.vectors[1] = torch::pow(parameters.vectors[1], -1);
	parameters.vectors[1] = torch::pow(parameters.vectors[1] + (parameters.vectors[1] * torch::square(outCinAccumulators) - 1.0f) * sigma_lr, -1);
#endif

	// modulate epsilons
#ifdef MODULATED
	outCinAccumulators = outCinAccumulators * (torch::tanh(parameters.matrices[1].matmul(inCoutActivations)) + 1.0f) * .005f; // .005f is a lr
#else
	outCinAccumulators *= theta_b_lr;
#endif

	// theta -= theta_grad * theta_b_lr (theta_grad = f(x_inCout) * epsilon_out.transpose)
	parameters.matrices[0] -= torch::tanh(inCoutActivations).matmul(torch::transpose(outCinAccumulators, 0, 1));


	// bias -= bias_grad * theta_b_lr (bias_grad = epsilon_outCin)
	parameters.vectors[0] -= outCinAccumulators;


	for (int i = 0; i < nChildren; i++)
	{
		children[i].thetaUpdate_simultaneous();
	}


#else

	// f(X_inCout) is precomputed, stored in the grad accumulator for convenience
	inCoutAccumulators.noalias() = inCoutActivations.array().tanh().matrix();





	// stored in the grad accumulator for convenience: epsilon_out = X_out - (theta*f(X_inCout) + bias)    (* invSigmas_out #ifdef ACTIVATION_VARIANCE)
	outputAccumulators.noalias() = ((outputActivations - (toOutput.vectors[0] + toOutput.matrices[0] * inCoutAccumulators)
#ifdef ACTIVATION_VARIANCE
		).array() * toOutput.vectors[1].array()).matrix();
#else
		));
#endif

#ifdef ACTIVATION_VARIANCE
	// Reminder that InternalConnexion_P.vectors[1] contains the inverses of the variances.
	// sigma_out += (invSigmas_out * (  X_out - (theta*f(X_inCout)+bias)  )^2  - 1) * sigma_lr.
	toOutput.vectors[1] = toOutput.vectors[1].cwiseInverse();
	toOutput.vectors[1] = (toOutput.vectors[1] + (((toOutput.vectors[1].array() * outputAccumulators.array().square()) - 1.0f) * sigma_lr).matrix()).cwiseInverse();
#endif

	// modulate epsilons
#ifdef MODULATED
	outputAccumulators = outputAccumulators.array() * ((toOutput.matrices[1] * inCoutActivations).array().tanh() + 1.0f) * .005f;
#else
	outputAccumulators *= theta_b_lr;
#endif

	// theta -= theta_grad * theta_b_lr (theta_grad = f(x_inCout) * epsilon_out.transpose)
	toOutput.matrices[0] -= outputAccumulators * inCoutAccumulators.transpose();


	// bias -= bias_grad * theta_b_lr (bias_grad = epsilon_out)
	toOutput.vectors[0] -= outputAccumulators;








	for (int i = 0; i < nChildren; i++)
	{
		// stored in the grad accumulator for convenience: epsilon_cIn = X_cIn - (theta*f(X_inCout) + bias)    (* invSigmas_cIn #ifdef ACTIVATION_VARIANCE)
		children[i].inputAccumulators.noalias() = ((children[i].inputActivations - (toChildren[i].vectors[0] + toChildren[i].matrices[0] * inCoutAccumulators)
#ifdef ACTIVATION_VARIANCE
			).array() * toChildren[i].vectors[1].array()).matrix();
#else
			));
#endif

#ifdef ACTIVATION_VARIANCE
		// Reminder that InternalConnexion_P.vectors[1] contains the inverses of the variances.
		// sigma_Cin += (invSigmas_Cin * (  X_Cin - (theta*f(X_inCout)+bias)  )^2  - 1) * sigma_lr.
		toChildren[i].vectors[1] = toChildren[i].vectors[1].cwiseInverse();
		toChildren[i].vectors[1] = (toChildren[i].vectors[1] + (((toChildren[i].vectors[1].array() * children[i].inputAccumulators.array().square()) - 1.0f) * sigma_lr).matrix()).cwiseInverse();
#endif

		// modulate epsilons
#ifdef MODULATED
		children[i].inputAccumulators = children[i].inputAccumulators.array() * ((toChildren[i].matrices[1] * inCoutActivations).array().tanh() + 1.0f) * .005f;
#else
		children[i].inputAccumulators *= theta_b_lr;
#endif

		// theta -= theta_grad * theta_b_lr (theta_grad = f(x_inCout) * epsilon_cIn.transpose)
		toChildren[i].matrices[0] -= children[i].inputAccumulators * inCoutAccumulators.transpose();

		// bias -= bias_grad * theta_b_lr (bias_grad = epsilon_cIn)
		toChildren[i].vectors[0] -= children[i].inputAccumulators;




		// child's update
		children[i].thetaUpdate_simultaneous();
	}

#endif


	return;
}

