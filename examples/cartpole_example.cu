#include <mppi/instantiations/cartpole_mppi/cartpole_mppi.cuh>
#include <iostream>
#include <chrono>

// 定义采样器类型为高斯分布，模板参数为CartpoleDynamics的动态参数类型
using SAMPLER_T = mppi::sampling_distributions::GaussianDistribution<CartpoleDynamics::DYN_PARAMS_T>;

int main(int argc, char** argv)
{
  // 创建倒立摆模型实例，参数分别为质量、长度等（此处均设为1.0）
  auto model = new CartpoleDynamics(1.0, 1.0, 1.0);

  // 创建倒立摆二次代价函数实例，用于评估控制策略的代价
  auto cost = new CartpoleQuadraticCost;

  // 设置控制量的范围（这里假设控制量为二维，x轴范围[-5, 5]）
  model->control_rngs_->x = -5;
  model->control_rngs_->y = 5;

  // 创建并设置新的代价函数参数
  CartpoleQuadraticCostParams new_params;
  new_params.cart_position_coeff = 50;              // 小车位置代价系数
  new_params.pole_angle_coeff = 200;                // 杆角代价系数
  new_params.cart_velocity_coeff = 10;              // 小车速度代价系数
  new_params.pole_angular_velocity_coeff = 1;       // 杆角速度代价系数
  new_params.control_cost_coeff[0] = 0;             // 控制代价系数（此处设为0）
  new_params.terminal_cost_coeff = 0;               // 终端状态代价系数（此处设为0）
  // 设置期望的终端状态：小车位置20，小车速度0，杆角为π（竖直向上），杆角速度0
  new_params.desired_terminal_state[0] = 20;
  new_params.desired_terminal_state[1] = 0;
  new_params.desired_terminal_state[2] = M_PI;
  new_params.desired_terminal_state[3] = 0;

  // 将新的代价参数传递给代价函数实例
  cost->setParams(new_params);

  // 设置仿真参数
  float dt = 0.02;         // 时间步长
  int max_iter = 1;        // 最大迭代次数
  float lambda = 0.25;     // 正则化参数
  float alpha = 0.0;       // 探索参数
  const int num_timesteps = 100; // 控制序列中的时间步数

  // 设置高斯分布采样器的参数
  auto sampler_params = SAMPLER_T::SAMPLING_PARAMS_T();
  for (int i = 0; i < CartpoleDynamics::CONTROL_DIM; i++)
  {
    sampler_params.std_dev[i] = 5.0; // 每个控制维度的标准差设为5.0
  }
  // 根据设置的参数创建采样器实例
  auto sampler = new SAMPLER_T(sampler_params);

  // 创建反馈控制器（DDP反馈控制器），用于计算控制律
  auto fb_controller = new DDPFeedback<CartpoleDynamics, num_timesteps>(model, dt);

  // 创建Vanilla MPPI控制器实例，传入模型、代价函数、反馈控制器、采样器及其它参数
  auto CartpoleController =
      new VanillaMPPIController<CartpoleDynamics, CartpoleQuadraticCost, DDPFeedback<CartpoleDynamics, num_timesteps>,
                                num_timesteps, 2048>(model, cost, fb_controller, sampler, dt, max_iter, lambda, alpha);

  // 获取控制器的参数，并设置仿真中并行rollout的维度
  auto controller_params = CartpoleController->getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 4, 1);
  controller_params.cost_rollout_dim_ = dim3(64, 4, 1);
  CartpoleController->setParams(controller_params);

  // 初始化状态、下一个状态和输出数组，全部置零
  CartpoleDynamics::state_array current_state = CartpoleDynamics::state_array::Zero();
  CartpoleDynamics::state_array next_state = CartpoleDynamics::state_array::Zero();
  CartpoleDynamics::output_array output = CartpoleDynamics::output_array::Zero();

  // 设置仿真时间步数（总步数为5000）
  int time_horizon = 5000;

  // 初始化状态导数数组，置零
  CartpoleDynamics::state_array xdot = CartpoleDynamics::state_array::Zero();

  // 记录仿真开始时间
  auto time_start = std::chrono::system_clock::now();
  for (int i = 0; i < time_horizon; ++i)
  {
    // 根据当前状态计算控制输入，1表示只计算一个控制步长
    CartpoleController->computeControl(current_state, 1);

    // 获取当前计算出的控制序列中第一个控制输入
    CartpoleDynamics::control_array control;
    control = CartpoleController->getControlSeq().block(0, 0, CartpoleDynamics::CONTROL_DIM, 1);

    // 对当前状态和控制输入执行约束检查（如边界条件等）
    model->enforceConstraints(current_state, control);

    // 使用模型进行状态更新，计算下一个状态及状态导数
    model->step(current_state, next_state, xdot, control, output, i, dt);

    // 更新当前状态为下一个状态
    current_state = next_state;

    // 每50个时间步打印当前时间、基线代价以及当前状态信息
    if (i % 50 == 0)
    {
      printf("Current Time: %f    ", i * dt);
      printf("Current Baseline Cost: %f    ", CartpoleController->getBaselineCost());
      model->printState(current_state.data());
      //      std::cout << control << std::endl;
    }

    // 将控制序列滑动（舍弃已用的控制，准备下一次优化）
    CartpoleController->slideControlSequence(1);
  }
  // 记录仿真结束时间，并计算总耗时（毫秒）
  auto time_end = std::chrono::system_clock::now();
  auto diff = std::chrono::duration<double, std::milli>(time_end - time_start);
  printf("The elapsed time is: %f milliseconds\n", diff.count());
  //    std::cout << "The current control at timestep " << i << " is: " << CartpoleController.get_control_seq()[i] <<
  //    std::endl;

  // 释放动态分配的内存，防止内存泄漏
  delete (CartpoleController);
  delete (cost);
  delete (model);
  delete (fb_controller);
  delete sampler;

  return 0;
}
