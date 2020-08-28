#pragma once

#include<torch/torch.h>

namespace nn = torch::nn;

//
// the Graph Convolution layer
//
struct GraphConvImpl : nn::Module {
  GraphConvImpl(int64_t in_features, int64_t out_features, bool bias=true)
    : in_features_(in_features),
      out_features_(out_features),
      weight_(register_parameter("weight", torch::randn({in_features, out_features})))
  {
    if (bias) {
      bias_ = register_parameter("bias", torch::randn(out_features));
    }
    else {
      // TODO
      // bias_ = torch::Tensor(0);
    }
  }

  torch::Tensor forward(const torch::Tensor &input, const torch::Tensor &adj) {
    auto support = torch::mm(input, weight_);
    auto output  = torch::mm(adj, support);
    if (true) {
      return output + bias_;
    }
    else {
      return output;
    }
  }

  int64_t in_features_, out_features_;
  torch::Tensor weight_, bias_;  
};

TORCH_MODULE(GraphConv);
