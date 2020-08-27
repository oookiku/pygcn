#include<torch/torch.h>
#include<iostream>
#include<fstream>
// #include"utils.hpp"

namespace nn = torch::nn;
namespace F  = torch::nn::functional;

//
// definision of the Graph Convolution layer
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




void dispAdjMatrix(int n, torch::Tensor &adj)
{
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << (adj[i][j]).toType(torch::kInt64) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << adj << std::endl;
}

void addEdge(int64_t u, int64_t v, torch::Tensor &adj)
{
  adj[u-1][v-1] = 1;
  adj[v-1][u-1] = 1;
}

torch::Tensor normalize(const torch::Tensor &mx)
{
  auto rowsum    = torch::sum(mx, 1);
  auto r_inv     = torch::pow(rowsum, -1.);
  auto r_mat_inv = torch::diag(r_inv);
  auto mx_hat    = torch::mm(r_mat_inv, mx.to(torch::kFloat32));
  return mx_hat;
}

torch::Tensor encodeOnehot(const torch::Tensor &label)
{



}


int main() {
  // making adjacency_matrix
  std::ifstream file("../data/karate_edges_77.txt");

  if (!file) {
    std::cout << "Cannot open the file!" << std::endl;
  }

  constexpr int n = 34;
  auto A = torch::zeros({n, n}, torch::kInt64);
  int64_t u, v;
  while(file >> u >> v) {
    addEdge(u, v, A);
  }

  auto A_tilde = A + torch::eye(n).to(torch::kInt64);  
  auto A_hat   = normalize(std::move(A_tilde));

  auto labels = torch::zeros(n).to(torch::kInt64);

  auto X = torch::eye(n).to(torch::kFloat32);

  auto idx_train = torch::arange( 0, 10, 1).to(torch::kInt64);
  auto idx_val   = torch::arange(10, 20, 1).to(torch::kInt64);
  auto idx_test  = torch::arange(20, 34, 1).to(torch::kInt64);

  //
  // initial settings
  //
  const int64_t n_node    = 34;
  const int64_t n_feature = 34;
  const int64_t n_hidden  = 16;
  const int64_t n_class   =  4; 

  const int64_t n_epoch   = 100; 

  const float lr    = 0.01;
  const float beta1 = 5e-4;

  const float pdropout = 0.5;
 
  //
  // model and optimizer
  //
  auto model = GCN(n_feature,
                   n_hidden,
                   n_class,
                   pdropout);

  auto optimizer = torch::optim::Adam(model->parameters(),
                                      torch::optim::AdamOptions(lr).weight_decay(beta1));

  //
  // device settings
  //
  auto device = torch::kCPU;
  std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! " << std::endl;
    device = torch::kCUDA;
  }

/*
  //
  // training loop
  //
  for (int64_t epoch = 1; epoch <= n_epoch; ++epoch) {
    model->train();
    optimizer->zero_grad();
    auto output = gcn->forward(X, A_hat); 
    float loss_train = loss(output.index(idx_train), labels.index(idx_train));
    float acc_train  = accuracy(output.index(idx_train), labels.index(idx_train));
    loss_train.backward();
    optimizer.step();

    float loss_val = loss(output.index(idx_val), labels.index(idx_val));
    float acc_val  = accuracy(output.index(idx_val), labels.index(idx_val)); 
  }
*/

/*
  //
  // testing
  //
  model->eval();
  auto output = model();
  loss_test   = 
  acc_test    = 
*/
}
