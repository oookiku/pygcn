#include<torch/torch.h>
#include<iostream>
#include<fstream>
// #include"utils.hpp"
#include "models.hpp"

namespace nn = torch::nn;
namespace F  = torch::nn::functional;

inline void addEdge(int64_t u, int64_t v, torch::Tensor &adj)
{
  adj[u-1][v-1] = 1;
  adj[v-1][u-1] = 1;
}

inline torch::Tensor normalize(const torch::Tensor &mx)
{
  auto rowsum    = torch::sum(mx, 1);
  auto r_inv     = torch::pow(rowsum, -1.);
  auto r_mat_inv = torch::diag(r_inv);
  auto mx_hat    = torch::mm(r_mat_inv, mx.to(torch::kFloat32));
  return mx_hat;
}

inline torch::Tensor encodeOnehot(int n_class, const torch::Tensor &labels)
{
  auto onehot = torch::eye(n_class).index(labels);
  return onehot;
}

torch::Tensor accuracy(const torch::Tensor &output, const torch::Tensor &labels)
{
  constexpr int idx = 1;
  constexpr int col = 0;
  constexpr int row = 1;

  auto preds = std::get<idx>(output.max(row)).type_as(labels);
  auto correct = preds.eq(labels);
  auto correct_sum = correct.sum().to(torch::kFloat64);
  double col_len = labels.size(col);
  auto len = torch::tensor(col_len).to(torch::kFloat64);
  return correct_sum / len;
}


int main() {
  //
  // initial settings
  //
  const int64_t n_node    = 34;
  const int64_t n_feature = 34;
  const int64_t n_hidden  = 16;
  const int64_t n_class   =  2; 

  const int64_t n_epoch   = 500; 

  const float lr    = 0.01;
  const float beta1 = 5e-4;

  const float pdropout = 0.5;


  //
  // adjacency_matrix & feature matrix & lables
  //
  std::ifstream file1("../data/karate_edges_77.txt");

  if (!file1) {
    std::cout << "Cannot open the file1 !" << std::endl;
  }

  constexpr int n = 34;
  auto A = torch::zeros({n, n}, torch::kInt64);
  int64_t u, v;
  while(file1 >> u >> v) {
    addEdge(u, v, A);
  }

  auto A_tilde = A + torch::eye(n).to(torch::kInt64);  
  auto A_hat   = normalize(std::move(A_tilde));

  std::ifstream file2("../data/karate_groups.txt");

  if (!file2) {
    std::cout << "Cannot open the file2 !" << std::endl;
  }

  auto labels = torch::zeros(n).to(torch::kInt64);

  int64_t id, group;
  while(file2 >> id >> group) {
    labels[id-1] = group-1;
  }

  auto X = torch::eye(n).to(torch::kFloat32);

  auto idx_train = torch::arange( 0, 17, 1).to(torch::kInt64);
  auto idx_val   = torch::arange(17, 34, 1).to(torch::kInt64);
  auto idx_test  = torch::arange(20, 34, 1).to(torch::kInt64);

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


  //
  // training loop
  //
  for (int64_t epoch = 1; epoch <= n_epoch; ++epoch) {
    model->train();
    optimizer.zero_grad();
    auto output = model->forward(X, A_hat); 
    auto loss_train = F::nll_loss(output.index(idx_train), labels.index(idx_train));
    auto acc_train = accuracy(output.index(idx_train), labels.index(idx_train));
    loss_train.backward();
    optimizer.step();

    auto loss_val = F::nll_loss(output.index(idx_val), labels.index(idx_val));
    auto acc_val = accuracy(output.index(idx_val), labels.index(idx_val));


    std::cout << "loss_train : " << loss_train.item<float>() << std::endl;
    std::cout << "acc_train  : " << acc_train.item<double>() << std::endl; 
    std::cout << "loss_val   : " << loss_val.item<float>()   << std::endl;
    std::cout << "acc_val    : " << acc_val.item<double>()   << std::endl;
  }
}
