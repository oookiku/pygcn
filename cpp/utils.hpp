#pragma once

#include<torch/torch.h>

namespace F = torch::nn::functional;

inline void addEdge(const int64_t u, 
                    const int64_t v,
                    torch::Tensor &adj)
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

inline torch::Tensor encodeOnehot(const int n_class,
                                  const torch::Tensor &labels)
{
  auto onehot = torch::eye(n_class).index(labels);
  return onehot;
}

inline torch::Tensor accuracy(const torch::Tensor &output, 
                              const torch::Tensor &labels)
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

inline std::tuple<torch::Tensor,
                  torch::Tensor,
                  torch::Tensor,
                  torch::Tensor,
                  torch::Tensor>
loadKarateClub(torch::Device device=torch::kCPU)
{
  // data loading : edges & labels
  std::string fname1 = "../data/karate_edges_77.txt";
  std::ifstream file1(fname1);

  if (!file1) {
    std::cout << "Cannot open " << fname1 << std::endl;
  }

  std::string fname2 = "../data/karate_groups.txt";
  std::ifstream file2(fname2);

  if (!file2) {
    std::cout << "Cannot open " << fname2 << std::endl;
  }


  // adjacency matrix
  constexpr int n = 34;
  auto A = torch::zeros({n, n}, torch::kInt64);
  int64_t u, v;
  while(file1 >> u >> v) {
    addEdge(u, v, A);
  }

  auto A_tilde = A + torch::eye(n).to(torch::kInt64); 
  auto A_hat   = normalize(std::move(A_tilde)); 


  // labels
  auto labels = torch::zeros(n).to(torch::kInt64);
  int64_t id, group;
  while(file2 >> id >> group) {
    labels[id-1] = group-1;
  }


  // feature matrix
  auto X = torch::eye(n).to(torch::kFloat32);


  // index for test and validation
  auto idx_train = torch::arange( 0, 17, 1).to(torch::kInt64);
  auto idx_val   = torch::arange(17, 34, 1).to(torch::kInt64);

  return {A_hat.to(device), X.to(device), labels.to(device), idx_train.to(device), idx_val.to(device)};
}
