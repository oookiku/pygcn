#pragma once

#include<torch/torch.h>
#include"layers.hpp"

namespace nn = torch::nn;
namespace F  = torch::nn::functional;

//
// definition of the Graph Convolutional Networks
//
struct GCNImpl : nn::Module {
  GCNImpl(int64_t nfeat, int64_t nhid, int64_t nclass, float pdrop=0.5)
    : gc1_(register_module("gc1", GraphConv(nfeat, nhid))),
      gc2_(register_module("gc2", GraphConv(nhid, nclass))),
      pdrop_(pdrop)
  {}

  torch::Tensor forward(const torch::Tensor &x, const torch::Tensor &adj) {
    auto xtilde = F::relu(gc1_(x, adj));
    xtilde = F::dropout(xtilde, F::DropoutFuncOptions().p(pdrop_));
    xtilde = gc2_(xtilde, adj);
    return F::log_softmax(xtilde, F::LogSoftmaxFuncOptions(1));
  }

  float pdrop_;
  GraphConv gc1_, gc2_;
};

TORCH_MODULE(GCN);
