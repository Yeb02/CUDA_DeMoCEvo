#pragma once


#pragma warning( push, 0 )

#include <torch/torch.h>

#pragma warning( pop ) 



struct MatrixGenerator : torch::nn::Module {
    MatrixGenerator(int seedSize, int nColumns, int nRows, int kDim)
    {
        fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(seedSize, 128))); //.bias(false) ?
        fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(128, 128)));
        fc3 = register_module("fc3", torch::nn::Linear(torch::nn::LinearOptions(128, 128)));
        fc4 = register_module("fc4", torch::nn::Linear(torch::nn::LinearOptions(128, (nColumns + nRows) * kDim)));

        torch::NoGradGuard no_grad;

        for (auto& p : named_parameters()) {
            std::string y = p.key();
            auto z = p.value(); // note that z is a Tensor, same as &p : layers->parameters

            if (y.compare(2, 6, "weight") == 0) {
                torch::nn::init::kaiming_normal_(z);
                //torch::nn::init::xavier_normal_(z);
            }
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::tanh(fc1->forward(x));
        x = torch::tanh(fc2->forward(x));
        x = torch::tanh(fc3->forward(x));
        x = torch::tanh(fc4->forward(x));
        return x.contiguous();
    }

    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr }, fc4{ nullptr };
};

struct VectorGenerator : torch::nn::Module {

    VectorGenerator(int seedSize, int outVecSize)
    {
        fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(seedSize, 64)));
        fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(64, 64)));
        fc3 = register_module("fc3", torch::nn::Linear(torch::nn::LinearOptions(64, 64)));
        fc4 = register_module("fc4", torch::nn::Linear(torch::nn::LinearOptions(64, outVecSize)));

        torch::NoGradGuard no_grad;

        for (auto& p : named_parameters()) {
            std::string y = p.key();
            auto z = p.value(); // note that z is a Tensor, same as &p : layers->parameters

            if (y.compare(2, 6, "weight") == 0) {
                torch::nn::init::kaiming_normal_(z);
                //torch::nn::init::xavier_normal_(z);
            }
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::tanh(fc1->forward(x));
        x = torch::tanh(fc2->forward(x));
        x = torch::tanh(fc3->forward(x));
        x = torch::tanh(fc4->forward(x));
        return x;
    }

    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr }, fc4{ nullptr };
};
