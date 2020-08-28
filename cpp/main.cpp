#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include "models.hpp"
#include "utils.hpp"

namespace F  = torch::nn::functional;

int main()
{
  // initial settings
  const int64_t n_node    = 34;
  const int64_t n_feature = 34;
  const int64_t n_hidden  = 16;
  const int64_t n_class   =  2; 

  const int64_t n_epoch   = 500; 

  const float lr    = 0.01;
  const float beta1 = 5e-4;

  const float pdropout = 0.5;


  // device settings
  auto device = torch::kCPU;
  std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! " << std::endl;
    device = torch::kCUDA;
  }


  // data loading
  auto [A_hat, X, labels, idx_train, idx_val] = loadKarateClub(device);


  // model and optimizer
  auto model = GCN(n_feature,
                   n_hidden,
                   n_class,
                   pdropout);
  model->to(device);

  auto optimizer = torch::optim::Adam(model->parameters(),
                                      torch::optim::AdamOptions(lr).weight_decay(beta1));


  // training loop
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
