#pragma once


#pragma warning( push, 0 )

#include <torch/torch.h>

#pragma warning( pop ) 




struct Matrixator : torch::nn::Module {
    Matrixator(int inSize, int nColumns, int nLines, int colEmbS, int lineEmbS) 
    {
        fc1 = register_module("fc1", torch::nn::Linear(inSize, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 64));
        fc4 = register_module("fc4", torch::nn::Linear(64, nColumns*colEmbS + nLines*lineEmbS));

        torch::NoGradGuard no_grad;

        for (auto& p : named_parameters()) {
            std::string y = p.key();
            auto z = p.value(); // note that z is a Tensor, same as &p : layers->parameters

            if (y.compare(2, 6, "weight") == 0)
                torch::nn::init::xavier_normal_(z);
            else if (y.compare(2, 4, "bias") == 0)
                z.zero_();
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

struct Specialist : torch::nn::Module {

    Specialist(int inS, int outS)
    {
        fc1 = register_module("fc1", torch::nn::Linear(inS, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, outS));

        torch::NoGradGuard no_grad;

        for (auto& p : named_parameters()) {
            std::string y = p.key();
            auto z = p.value(); // note that z is a Tensor, same as &p : layers->parameters

            if (y.compare(2, 6, "weight") == 0)
                torch::nn::init::xavier_normal_(z);
            else if (y.compare(2, 4, "bias") == 0)
                z.zero_();
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::tanh(fc1->forward(x));
        x = torch::tanh(fc2->forward(x));
        x = torch::tanh(fc3->forward(x));
        return x;
    }

    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
};
