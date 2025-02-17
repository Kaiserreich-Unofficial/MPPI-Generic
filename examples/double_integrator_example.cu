#include <mppi/dynamics/double_integrator/di_dynamics.cuh>         // 引入双积分器动力学模型头文件
#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>       // 引入二次代价函数头文件
#include <mppi/controllers/MPPI/mppi_controller.cuh>                  // 引入MPPI控制器头文件
#include <mppi/feedback_controllers/DDP/ddp.cuh>                       // 引入基于DDP的反馈控制器头文件

#include <mppi/sampling_distributions/colored_noise/colored_noise.cuh> // 引入有色噪声采样器头文件

#include <iomanip>  // 用于格式化输出

// 定义宏，启用有色噪声（若取消注释，则使用高斯噪声采样器）
#define USE_COLORED_NOISE

// 定义MPPI控制器的时间步数和采样数
const int TIMESTEPS = 65;
const int NUM_ROLLOUTS = 128;

// 类型别名定义：
// DYN 表示双积分器动力学模型
using DYN = DoubleIntegratorDynamics;
// COST 表示针对双积分器的二次代价函数
using COST = QuadraticCost<DYN>;
// FB_CONTROLLER 表示基于DDP的反馈控制器，其控制序列长度为 TIMESTEPS
using FB_CONTROLLER = DDPFeedback<DYN, TIMESTEPS>;

// 根据是否定义USE_COLORED_NOISE选择不同的采样器类型：
// - 启用有色噪声时使用 ColoredNoiseDistribution
// - 否则使用标准高斯分布采样器
#ifdef USE_COLORED_NOISE
using SAMPLER = mppi::sampling_distributions::ColoredNoiseDistribution<DYN::DYN_PARAMS_T>;
#else
using SAMPLER = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
#endif

int main()
{
  // 设置时间步长
  float dt = 0.015;

  // 设置初始状态：位置为(-9, -9)，速度为(0.1, 0.1)
  DYN::state_array x;
  x << -9, -9, 0.1, 0.1;
  DYN::state_array xdot;  // 用于存储状态变化率

  // 初始化采样器参数：设置控制噪声的标准差为0.5
  SAMPLER::SAMPLING_PARAMS_T sampler_params;
  for (int i = 0; i < DYN::CONTROL_DIM; i++)
  {
    sampler_params.std_dev[i] = 0.5;
#ifdef USE_COLORED_NOISE
    // 若使用有色噪声，则设置噪声指数（决定噪声的相关性）
    sampler_params.exponents[i] = 1.0;
#endif
  }
  // 创建采样器对象（根据定义使用有色噪声或高斯噪声）
  SAMPLER sampler(sampler_params);

  // 创建动力学模型、代价函数以及反馈控制器
  DYN model;
  COST cost;
  FB_CONTROLLER fb_controller = FB_CONTROLLER(&model, dt);

  // 获取并设置代价函数参数
  auto cost_params = cost.getParams();

  // 设置目标状态和权重参数
  DYN::state_array x_goal;
  x_goal << -4, -4, 0, 0;         // 目标状态（位置为(-4, -4)，速度为0）
  DYN::state_array q_coeffs;
  q_coeffs << 5, 5, 0.5, 0.5;      // 状态误差的权重
  for (int i = 0; i < DYN::STATE_DIM; i++)
  {
    cost_params.s_coeffs[i] = q_coeffs[i];  // 设定每个状态变量的权重
    cost_params.s_goal[i] = x_goal[i];        // 设定目标状态
  }
  cost.setParams(cost_params);

  // 创建Vanilla MPPI控制器
  float lambda = 1;              // 正则化参数或学习率
  float alpha = 1.0;             // 探索参数
  int max_iter = 1;              // 每个时间步内的最大迭代次数
  int total_time_horizon = 300;  // 仿真总步数

  auto controller = VanillaMPPIController<DYN, COST, FB_CONTROLLER, TIMESTEPS, NUM_ROLLOUTS, SAMPLER>(
      &model, &cost, &fb_controller, &sampler, dt, max_iter, lambda, alpha);

  // 配置控制器的rollout计算并行维度（动力学和代价函数的计算）
  auto controller_params = controller.getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 1, 1);
  controller_params.cost_rollout_dim_ = dim3(64, 1, 1);
  controller.setParams(controller_params);

  /********************** Vanilla MPPI 仿真 **********************/
  float cumulative_cost = 0;   // 用于累计总代价
  int crash = 0;               // 标记是否发生系统崩溃（例如违反约束）
  for (int t = 0; t < total_time_horizon; ++t)
  {
    // 根据当前状态计算控制输入
    controller.computeControl(x, 1);

    // 获取名义控制序列及自由能统计信息（自由能用于评估采样质量）
    auto nominal_control = controller.getControlSeq();
    auto fe_stat = controller.getFreeEnergyStatistics();

    // 选择当前控制输入（取控制序列中的第一个控制）
    DYN::control_array current_control = nominal_control.col(0);

    // 向前传播系统动力学，计算下一状态和输出
    DYN::output_array y;
    DYN::state_array x_next;
    // 通过step函数计算下一状态：传入当前状态、当前控制、时间步、dt等
    model.step(x, x_next, xdot, current_control, y, t, dt);
    x = x_next;  // 更新状态

    // 每隔10个时间步输出一次状态信息
    if (t % 10 == 0)
    {
      std::cout << "T: " << std::fixed << std::setprecision(3) << t * dt;
      // 可选：输出自由能信息（此处被注释掉）
      // << "s Free Energy: " << fe_stat.real_sys.freeEnergyMean
      // << " +- " << fe_stat.real_sys.freeEnergyVariance << std::endl;
      std::cout << " X: " << x.transpose() << std::endl;
    }

    // 将控制序列平移，为下一时间步准备
    controller.slideControlSequence(1);
    // 累计当前时间步的运行代价（传入输出、当前控制、时间步及崩溃标记）
    cumulative_cost += cost.computeRunningCost(y, current_control, t, &crash);
  }
  // 输出总累计代价
  std::cout << "Total Cost: " << cumulative_cost << std::endl;

  return 0;
}
