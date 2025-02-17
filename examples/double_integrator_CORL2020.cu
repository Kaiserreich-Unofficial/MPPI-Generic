// 引入Double Integrator（双积分器）动力学模型和相关代价函数、控制器及反馈控制器头文件
#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_circle_cost.cuh>
#include <mppi/cost_functions/double_integrator/double_integrator_robust_cost.cuh>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/controllers/Tube-MPPI/tube_mppi_controller.cuh>
#include <mppi/controllers/R-MPPI/robust_mppi_controller.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>

#include <cnpy.h>         // 用于保存数据到NumPy .npy文件
#include <random>         // 用于生成控制轨迹中的随机噪声

// 检查系统是否失败的函数，根据状态s（例如，二维位置）判断是否超出预定圆环区域
bool tubeFailure(float* s)
{
  float inner_path_radius2 = 1.675 * 1.675;  // 内部圆半径的平方
  float outer_path_radius2 = 2.325 * 2.325;  // 外部圆半径的平方
  float radial_position = s[0] * s[0] + s[1] * s[1];  // 当前状态的径向平方
  if ((radial_position < inner_path_radius2) || (radial_position > outer_path_radius2))
  {
    return true;  // 如果小于内径或大于外径，则认为系统失败
  }
  else
  {
    return false;
  }
}

// 定义常用别名和常量
using Dyn = DoubleIntegratorDynamics;         // 双积分器动力学模型类型
using SCost = DoubleIntegratorCircleCost;       // 标准圆形代价函数类型
using RCost = DoubleIntegratorRobustCost;       // 鲁棒代价函数类型
const int num_timesteps = 50;                   // 优化时间范围（控制序列长度）
const int total_time_horizon = 5000;            // 总仿真步数

using Feedback = DDPFeedback<Dyn, num_timesteps>;  // 基于DDP的反馈控制器
using Sampler = mppi::sampling_distributions::GaussianDistribution<Dyn::DYN_PARAMS_T>;  // 采用高斯分布采样器

// 动力学传播时间步长、最大迭代次数、学习率参数和探索参数
const float dt = 0.02;      // 时间步长
const int max_iter = 1;     // 优化最大迭代次数
const float lambda = 2;     // 正则化/学习率参数
const float alpha = 0.0;    // 探索参数

// 定义状态轨迹矩阵类型（状态维度 x 时间步数）
typedef Eigen::Matrix<float, Dyn::STATE_DIM, num_timesteps> state_trajectory;

// 将一段轨迹保存到std::vector中（按行保存，每个时间步内所有状态量）
void saveTraj(const Eigen::Ref<const state_trajectory>& traj, int t, std::vector<float>& vec)
{
  for (int i = 0; i < num_timesteps; i++)
  {
    for (int j = 0; j < Dyn::STATE_DIM; j++)
    {
      vec[t * num_timesteps * Dyn::STATE_DIM + i * Dyn::STATE_DIM + j] = traj(j, i);
    }
  }
}

// 将单个状态保存到std::vector中
void saveState(const Eigen::Ref<const Dyn::state_array>& state, int t, std::vector<float>& vec)
{
  for (int j = 0; j < Dyn::STATE_DIM; j++)
  {
    vec[t * Dyn::STATE_DIM + j] = state(j);
  }
}

// runVanilla：运行标准的Vanilla MPPI控制器仿真
void runVanilla(const Eigen::Ref<const Eigen::Matrix<float, Dyn::STATE_DIM, total_time_horizon>>& noise)
{
  // 设置初始状态：例如，x位置=2，速度=0，其他状态变量依次设定
  Dyn::state_array x;
  x << 2, 0, 0, 1;
  Dyn::state_array xdot;  // 状态导数

  // 设置控制噪声参数：每个控制维度的标准差设为1
  Sampler::SAMPLING_PARAMS_T sampler_params;
  for (int i = 0; i < Dyn::CONTROL_DIM; i++)
  {
    sampler_params.std_dev[i] = 1;
  }

  // 用于保存实际轨迹、名义轨迹和自由能（free energy）的数据容器
  std::vector<float> van_trajectory(Dyn::STATE_DIM * total_time_horizon, 0);
  std::vector<float> van_nominal_traj(Dyn::STATE_DIM * num_timesteps * total_time_horizon, 0);
  std::vector<float> van_free_energy(total_time_horizon, 0);

  // 初始化控制相关对象：模型、代价函数、采样器和反馈控制器
  Dyn model;
  SCost cost;
  Sampler sampler(sampler_params);

  // 初始化DDP反馈控制器，并设置其代价矩阵Q
  Feedback fb_controller(&model, dt);
  auto fb_params = fb_controller.getParams();
  fb_params.Q.diagonal() << 500, 500, 100, 100;
  fb_controller.setParams(fb_params);

  // 创建Vanilla MPPI控制器实例，模板参数指定动力学、代价、反馈、时间步数、采样数和采样器类型
  auto controller = VanillaMPPIController<Dyn, SCost, Feedback, num_timesteps, 1024, Sampler>(
      &model, &cost, &fb_controller, &sampler, dt, max_iter, lambda, alpha);

  // 配置控制器的rollout并行维度
  auto controller_params = controller.getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 1, 1);
  controller_params.cost_rollout_dim_ = dim3(64, 1, 1);
  controller.setParams(controller_params);

  // 初始化反馈控制相关数据
  controller.initFeedback();

  // 开始仿真循环
  for (int t = 0; t < total_time_horizon; ++t)
  {
    // 计算控制输入（单步计算）
    controller.computeControl(x, 1);

    // 计算反馈增益，用于修正开放环控制
    controller.computeFeedback(x);

    // 传播反馈控制下的状态序列（名义轨迹）
    controller.computeFeedbackPropagatedStateSeq();

    // 获取名义状态轨迹和控制序列，以及自由能统计信息
    auto nominal_trajectory = controller.getTargetStateSeq();
    auto nominal_control = controller.getControlSeq();
    auto fe_stat = controller.getFreeEnergyStatistics();

    // 保存当前状态和名义轨迹，同时记录自由能均值
    saveState(x, t, van_trajectory);
    saveTraj(nominal_trajectory, t, van_nominal_traj);
    van_free_energy[t] = fe_stat.real_sys.freeEnergyMean;

    // 取出开放环控制输入（名义控制序列的第一个控制）
    DoubleIntegratorDynamics::control_array current_control = nominal_control.col(0);

    // 计算反馈控制部分并叠加到开放环控制上
    Dyn::control_array fb_control = controller.getFeedbackControl(x, nominal_trajectory.col(0), 0);
    current_control += fb_control;

    // 利用当前控制计算状态导数，并更新状态（通过动力学传播）
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);

    // 加入系统噪声干扰（根据预先生成的噪声矩阵）
    x += noise.col(t) * sqrt(model.getParams().system_noise) * dt;

    // 控制序列平移，舍弃已经执行的控制，准备下一个时间步
    controller.slideControlSequence(1);
  }
  // 将数据保存到NumPy .npy文件中，便于后续分析
  cnpy::npy_save("vanilla_state_trajectory.npy", van_trajectory.data(),
                 { total_time_horizon, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("vanilla_nominal_trajectory.npy", van_nominal_traj.data(),
                 { total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("vanilla_free_energy.npy", van_free_energy.data(), { total_time_horizon }, "w");
}

// runVanillaLarge：运行大规模Vanilla MPPI控制器仿真（可能使用了更多参数或特殊配置）
void runVanillaLarge(const Eigen::Ref<const Eigen::Matrix<float, Dyn::STATE_DIM, total_time_horizon>>& noise)
{
  // 设置初始状态
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 1;
  DoubleIntegratorDynamics::state_array xdot;

  // 设置采样器参数（控制噪声标准差）
  Sampler::SAMPLING_PARAMS_T sampler_params;
  for (int i = 0; i < Dyn::CONTROL_DIM; i++)
  {
    sampler_params.std_dev[i] = 1;
  }

  // 用于保存状态轨迹、名义轨迹及自由能数据
  std::vector<float> van_large_trajectory(Dyn::STATE_DIM * total_time_horizon, 0);
  std::vector<float> van_large_nominal_traj(Dyn::STATE_DIM * num_timesteps * total_time_horizon, 0);
  std::vector<float> van_large_free_energy(total_time_horizon, 0);

  // 初始化控制器：注意此处传入了一个参数100给模型构造函数，可能表示仿真中的某个特殊参数（如质量或惯性）
  Dyn model(100);
  SCost cost;
  Sampler sampler(sampler_params);

  // 设置DDP反馈控制器参数
  Feedback fb_controller(&model, dt);
  auto fb_params = fb_controller.getParams();
  fb_params.Q.diagonal() << 500, 500, 100, 100;
  fb_controller.setParams(fb_params);

  // 创建Vanilla MPPI控制器
  auto controller = VanillaMPPIController<Dyn, SCost, Feedback, num_timesteps, 1024, Sampler>(
      &model, &cost, &fb_controller, &sampler, dt, max_iter, lambda, alpha);
  auto controller_params = controller.getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 1, 1);
  controller_params.cost_rollout_dim_ = dim3(64, 1, 1);
  controller.setParams(controller_params);
  controller.initFeedback();

  // 开始仿真循环
  for (int t = 0; t < total_time_horizon; ++t)
  {
    /********************** Vanilla Large **********************/
    controller.computeControl(x, 1);
    controller.computeFeedback(x);
    controller.computeFeedbackPropagatedStateSeq();

    auto nominal_trajectory = controller.getTargetStateSeq();
    auto nominal_control = controller.getControlSeq();
    auto fe_stat = controller.getFreeEnergyStatistics();

    // 保存当前状态、名义轨迹及自由能
    saveState(x, t, van_large_trajectory);
    saveTraj(nominal_trajectory, t, van_large_nominal_traj);
    van_large_free_energy[t] = fe_stat.real_sys.freeEnergyMean;

    // 取开放环控制并叠加反馈控制
    DoubleIntegratorDynamics::control_array current_control = nominal_control.col(0);
    Dyn::control_array fb_control = controller.getFeedbackControl(x, nominal_trajectory.col(0), 0);
    current_control += fb_control;

    // 状态传播和更新
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);

    // 添加噪声干扰
    x += noise.col(t) * sqrt(model.getParams().system_noise) * dt;

    // 平移控制序列
    controller.slideControlSequence(1);
  }
  // 保存数据
  cnpy::npy_save("vanilla_large_state_trajectory.npy", van_large_trajectory.data(),
                 { total_time_horizon, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("vanilla_large_nominal_trajectory.npy", van_large_nominal_traj.data(),
                 { total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("vanilla_large_free_energy.npy", van_large_free_energy.data(), { total_time_horizon }, "w");
}

// runVanillaLargeRC：运行大规模Vanilla MPPI控制器仿真（使用鲁棒代价函数）
// 主要区别在于采用RC（Robust Cost）版本的代价函数及相应的参数设置
void runVanillaLargeRC(const Eigen::Ref<const Eigen::Matrix<float, Dyn::STATE_DIM, total_time_horizon>>& noise)
{
  // 设置初始状态
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 1;
  DoubleIntegratorDynamics::state_array xdot;

  // 设置采样器参数
  Sampler::SAMPLING_PARAMS_T sampler_params;
  for (int i = 0; i < Dyn::CONTROL_DIM; i++)
  {
    sampler_params.std_dev[i] = 1;
  }

  // 初始化数据保存容器
  std::vector<float> van_large_trajectory(Dyn::STATE_DIM * total_time_horizon, 0);
  std::vector<float> van_large_nominal_traj(Dyn::STATE_DIM * num_timesteps * total_time_horizon, 0);
  std::vector<float> van_large_free_energy(total_time_horizon, 0);

  // 初始化模型（传入参数100）
  Dyn model(100);

  // 使用鲁棒代价函数
  RCost cost;
  Sampler sampler(sampler_params);
  auto params = cost.getParams();
  params.crash_cost = 100;
  cost.setParams(params);

  // 初始化DDP反馈控制器
  Feedback fb_controller(&model, dt);
  auto fb_params = fb_controller.getParams();
  fb_params.Q.diagonal() << 500, 500, 100, 100;
  fb_controller.setParams(fb_params);

  // 创建Vanilla MPPI控制器（鲁棒代价版本）
  auto controller = VanillaMPPIController<Dyn, RCost, Feedback, num_timesteps, 1024, Sampler>(
      &model, &cost, &fb_controller, &sampler, dt, max_iter, lambda, alpha);
  auto controller_params = controller.getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 1, 1);
  controller_params.cost_rollout_dim_ = dim3(64, 1, 1);
  controller.setParams(controller_params);
  controller.initFeedback();

  // 开始仿真循环
  for (int t = 0; t < total_time_horizon; ++t)
  {
    /********************** Vanilla Large with Robust Cost **********************/
    controller.computeControl(x, 1);
    controller.computeFeedback(x);
    controller.computeFeedbackPropagatedStateSeq();

    auto nominal_trajectory = controller.getTargetStateSeq();
    auto nominal_control = controller.getControlSeq();
    auto fe_stat = controller.getFreeEnergyStatistics();

    // 保存数据
    saveState(x, t, van_large_trajectory);
    saveTraj(nominal_trajectory, t, van_large_nominal_traj);
    van_large_free_energy[t] = fe_stat.real_sys.freeEnergyMean;

    // 获取并修正控制
    DoubleIntegratorDynamics::control_array current_control = nominal_control.col(0);
    Dyn::control_array fb_control = controller.getFeedbackControl(x, nominal_trajectory.col(0), 0);
    current_control += fb_control;

    // 状态传播和更新
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);

    // 添加扰动
    x += noise.col(t) * sqrt(model.getParams().system_noise) * dt;

    // 平移控制序列
    controller.slideControlSequence(1);
  }
  // 保存数据至文件
  cnpy::npy_save("vanilla_large_robust_state_trajectory.npy", van_large_trajectory.data(),
                 { total_time_horizon, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("vanilla_large_robust_nominal_trajectory.npy", van_large_nominal_traj.data(),
                 { total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("vanilla_large_robust_free_energy.npy", van_large_free_energy.data(), { total_time_horizon }, "w");
}

// runTube：运行Tube MPPI控制器仿真（标准代价版本）
void runTube(const Eigen::Ref<const Eigen::Matrix<float, Dyn::STATE_DIM, total_time_horizon>>& noise)
{
  // 设置初始状态
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 1;
  DoubleIntegratorDynamics::state_array xdot;

  // 设置采样器参数
  Sampler::SAMPLING_PARAMS_T sampler_params;
  for (int i = 0; i < Dyn::CONTROL_DIM; i++)
  {
    sampler_params.std_dev[i] = 1;
  }

  // 数据保存容器
  std::vector<float> tube_trajectory(Dyn::STATE_DIM * total_time_horizon, 0);
  std::vector<float> tube_nominal_traj(Dyn::STATE_DIM * num_timesteps * total_time_horizon, 0);
  std::vector<float> tube_nominal_free_energy(total_time_horizon, 0);
  std::vector<float> tube_real_free_energy(total_time_horizon, 0);
  std::vector<float> tube_nominal_state_used(total_time_horizon, 0);

  // 初始化模型、代价函数、采样器和DDP反馈控制器
  Dyn model(100);
  SCost cost;
  Sampler sampler(sampler_params);
  Feedback fb_controller(&model, dt);
  auto fb_params = fb_controller.getParams();
  fb_params.Q.diagonal() << 500, 500, 100, 100;
  fb_controller.setParams(fb_params);

  // 创建Tube MPPI控制器
  auto controller = TubeMPPIController<Dyn, SCost, Feedback, num_timesteps, 1024, Sampler>(
      &model, &cost, &fb_controller, &sampler, dt, max_iter, lambda, alpha);
  auto controller_params = controller.getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 1, 1);
  controller_params.cost_rollout_dim_ = dim3(64, 1, 1);
  controller.setParams(controller_params);

  // 设置名义状态阈值，用于判断控制器是否需要重新规划
  controller.setNominalThreshold(20);

  // 开始仿真循环
  for (int t = 0; t < total_time_horizon; ++t)
  {
    /********************** Tube **********************/
    controller.computeControl(x, 1);
    controller.computeFeedback(x);
    controller.computeFeedbackPropagatedStateSeq();

    auto nominal_trajectory = controller.getTargetStateSeq();
    auto nominal_control = controller.getControlSeq();
    auto fe_stat = controller.getFreeEnergyStatistics();

    // 保存当前状态、名义轨迹、自由能以及使用的名义状态信息
    saveState(x, t, tube_trajectory);
    saveTraj(nominal_trajectory, t, tube_nominal_traj);
    tube_nominal_free_energy[t] = fe_stat.nominal_sys.freeEnergyMean;
    tube_real_free_energy[t] = fe_stat.real_sys.freeEnergyMean;
    tube_nominal_state_used[t] = fe_stat.nominal_state_used;

    // 获取开放环控制，并计算反馈修正
    DoubleIntegratorDynamics::control_array current_control = nominal_control.col(0);
    Dyn::control_array fb_control = controller.getFeedbackControl(x, nominal_trajectory.col(0), 0);
    current_control += fb_control;

    // 状态传播和更新
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);
    // 更新名义状态（用于Tube控制器的状态束）
    controller.updateNominalState(current_control);

    // 添加噪声干扰
    x += noise.col(t) * sqrt(model.getParams().system_noise) * dt;

    // 平移控制序列
    controller.slideControlSequence(1);
  }
  // 保存数据
  cnpy::npy_save("tube_state_trajectory.npy", tube_trajectory.data(),
                 { total_time_horizon, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("tube_nominal_trajectory.npy", tube_nominal_traj.data(),
                 { total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("tube_nominal_free_energy.npy", tube_nominal_free_energy.data(), { total_time_horizon }, "w");
  cnpy::npy_save("tube_real_free_energy.npy", tube_real_free_energy.data(), { total_time_horizon }, "w");
  cnpy::npy_save("tube_nominal_state_used.npy", tube_nominal_state_used.data(), { total_time_horizon }, "w");
}

// runTubeRC：运行Tube MPPI控制器仿真（采用鲁棒代价函数版本）
void runTubeRC(const Eigen::Ref<const Eigen::Matrix<float, Dyn::STATE_DIM, total_time_horizon>>& noise)
{
  // 设置初始状态
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 1;
  DoubleIntegratorDynamics::state_array xdot;

  // 设置采样器参数
  Sampler::SAMPLING_PARAMS_T sampler_params;
  for (int i = 0; i < Dyn::CONTROL_DIM; i++)
  {
    sampler_params.std_dev[i] = 1;
  }

  // 数据保存容器
  std::vector<float> tube_trajectory(Dyn::STATE_DIM * total_time_horizon, 0);
  std::vector<float> tube_nominal_traj(Dyn::STATE_DIM * num_timesteps * total_time_horizon, 0);
  std::vector<float> tube_nominal_free_energy(total_time_horizon, 0);
  std::vector<float> tube_real_free_energy(total_time_horizon, 0);
  std::vector<float> tube_nominal_state_used(total_time_horizon, 0);

  // 初始化模型、鲁棒代价函数、采样器及DDP反馈控制器
  Dyn model(100);
  RCost cost;
  Sampler sampler(sampler_params);
  auto params = cost.getParams();
  params.crash_cost = 100;
  cost.setParams(params);
  Feedback fb_controller(&model, dt);
  auto fb_params = fb_controller.getParams();
  fb_params.Q.diagonal() << 500, 500, 100, 100;
  fb_controller.setParams(fb_params);

  // 创建Tube MPPI控制器（鲁棒代价版本）
  auto controller = TubeMPPIController<Dyn, RCost, Feedback, num_timesteps, 1024>(
      &model, &cost, &fb_controller, &sampler, dt, max_iter, lambda, alpha);
  auto controller_params = controller.getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 1, 1);
  controller_params.cost_rollout_dim_ = dim3(64, 1, 1);
  controller.setParams(controller_params);

  // 设置名义状态阈值
  controller.setNominalThreshold(2);

  // 开始仿真循环
  for (int t = 0; t < total_time_horizon; ++t)
  {
    /********************** Tube (鲁棒代价) **********************/
    controller.computeControl(x, 1);
    controller.computeFeedback(x);
    controller.computeFeedbackPropagatedStateSeq();

    auto nominal_trajectory = controller.getTargetStateSeq();
    auto nominal_control = controller.getControlSeq();
    auto fe_stat = controller.getFreeEnergyStatistics();

    // 保存数据
    saveState(x, t, tube_trajectory);
    saveTraj(nominal_trajectory, t, tube_nominal_traj);
    tube_nominal_free_energy[t] = fe_stat.nominal_sys.freeEnergyMean;
    tube_real_free_energy[t] = fe_stat.real_sys.freeEnergyMean;
    tube_nominal_state_used[t] = fe_stat.nominal_state_used;

    // 获取并修正控制输入
    DoubleIntegratorDynamics::control_array current_control = nominal_control.col(0);
    Dyn::control_array fb_control = controller.getFeedbackControl(x, nominal_trajectory.col(0), 0);
    current_control += fb_control;

    // 状态传播和更新，同时更新名义状态
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);
    controller.updateNominalState(current_control);

    // 添加噪声干扰
    x += noise.col(t) * sqrt(model.getParams().system_noise) * dt;

    // 平移控制序列
    controller.slideControlSequence(1);
  }
  // 保存数据
  cnpy::npy_save("tube_robust_state_trajectory.npy", tube_trajectory.data(),
                 { total_time_horizon, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("tube_robust_nominal_trajectory.npy", tube_nominal_traj.data(),
                 { total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("tube_robust_nominal_free_energy.npy", tube_nominal_free_energy.data(), { total_time_horizon }, "w");
  cnpy::npy_save("tube_robust_real_free_energy.npy", tube_real_free_energy.data(), { total_time_horizon }, "w");
  cnpy::npy_save("tube_robust_nominal_state_used.npy", tube_nominal_state_used.data(), { total_time_horizon }, "w");
}

// runRobustSc：运行基于RMPPI（鲁棒MPPI）控制器仿真（使用标准代价版本）
// 此函数中还计算了自由能上下界等指标
void runRobustSc(const Eigen::Ref<const Eigen::Matrix<float, Dyn::STATE_DIM, total_time_horizon>>& noise)
{
  // 设置初始状态
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 1;
  DoubleIntegratorDynamics::state_array xdot;

  // 设置采样器参数
  Sampler::SAMPLING_PARAMS_T sampler_params;
  for (int i = 0; i < Dyn::CONTROL_DIM; i++)
  {
    sampler_params.std_dev[i] = 1;
  }

  // 数据保存容器
  std::vector<float> robust_sc_trajectory(Dyn::STATE_DIM * total_time_horizon, 0);
  std::vector<float> robust_sc_nominal_traj(Dyn::STATE_DIM * num_timesteps * total_time_horizon, 0);
  std::vector<float> robust_sc_nominal_free_energy(total_time_horizon, 0);
  std::vector<float> robust_sc_real_free_energy(total_time_horizon, 0);
  std::vector<float> robust_sc_nominal_free_energy_bound(total_time_horizon, 0);
  std::vector<float> robust_sc_real_free_energy_bound(total_time_horizon, 0);
  std::vector<float> robust_sc_real_free_energy_growth_bound(total_time_horizon, 0);
  std::vector<float> robust_sc_nominal_state_used(total_time_horizon, 0);

  // 初始化控制器相关对象
  Dyn model(100);
  SCost cost;
  Sampler sampler(sampler_params);
  Feedback fb_controller(&model, dt);
  auto fb_params = fb_controller.getParams();
  fb_params.Q.diagonal() << 500, 500, 100, 100;
  fb_controller.setParams(fb_params);

  // 设置值函数阈值，用于约束自由能增长
  float value_function_threshold = 20.0;
  auto controller = RobustMPPIController<Dyn, SCost, Feedback, num_timesteps, 1024, Sampler>(
      &model, &cost, &fb_controller, &sampler, dt, max_iter, lambda, alpha, value_function_threshold);
  auto controller_params = controller.getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 1, 1);
  controller_params.cost_rollout_dim_ = dim3(64, 1, 1);
  controller.setParams(controller_params);

  // 开始仿真循环
  for (int t = 0; t < total_time_horizon; ++t)
  {
    /********************** Robust (Standard Cost) **********************/
    // 更新重要性采样控制（用于改进采样效果），再计算控制
    controller.updateImportanceSamplingControl(x, 1);
    controller.computeControl(x, 1);
    controller.computeFeedback(x);
    controller.computeFeedbackPropagatedStateSeq();

    auto nominal_trajectory = controller.getTargetStateSeq();
    auto nominal_control = controller.getControlSeq();
    auto fe_stat = controller.getFreeEnergyStatistics();

    // 保存当前状态、名义轨迹和自由能数据，同时计算自由能界
    saveState(x, t, robust_sc_trajectory);
    saveTraj(nominal_trajectory, t, robust_sc_nominal_traj);
    robust_sc_nominal_free_energy[t] = fe_stat.nominal_sys.freeEnergyMean;
    robust_sc_real_free_energy[t] = fe_stat.real_sys.freeEnergyMean;
    robust_sc_nominal_free_energy_bound[t] =
        value_function_threshold + 2 * fe_stat.nominal_sys.freeEnergyModifiedVariance;
    robust_sc_real_free_energy_bound[t] = 0;
    robust_sc_real_free_energy_growth_bound[t] = 0;
    robust_sc_nominal_state_used[t] = fe_stat.nominal_state_used;

    // 取出并修正当前控制
    DoubleIntegratorDynamics::control_array current_control = nominal_control.col(0);
    Dyn::control_array fb_control = controller.getFeedbackControl(x, nominal_trajectory.col(0), 0);
    current_control += fb_control;

    // 状态传播和更新
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);

    // 添加扰动噪声
    x += noise.col(t) * sqrt(model.getParams().system_noise) * dt;

    // 平移控制序列
    controller.slideControlSequence(1);
  }
  // 保存数据
  cnpy::npy_save("robust_sc_state_trajectory.npy", robust_sc_trajectory.data(),
                 { total_time_horizon, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("robust_sc_nominal_trajectory.npy", robust_sc_nominal_traj.data(),
                 { total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("robust_sc_nominal_free_energy.npy", robust_sc_nominal_free_energy.data(), { total_time_horizon },
                 "w");
  cnpy::npy_save("robust_sc_real_free_energy.npy", robust_sc_real_free_energy.data(), { total_time_horizon }, "w");
  cnpy::npy_save("robust_sc_nominal_state_used.npy", robust_sc_nominal_state_used.data(), { total_time_horizon }, "w");
  cnpy::npy_save("robust_sc_real_free_energy_bound.npy", robust_sc_nominal_free_energy_bound.data(),
                 { total_time_horizon }, "w");
  cnpy::npy_save("robust_sc_nominal_free_energy_bound.npy", robust_sc_real_free_energy_bound.data(),
                 { total_time_horizon }, "w");
  cnpy::npy_save("robust_sc_real_free_energy_growth_bound.npy", robust_sc_real_free_energy_growth_bound.data(),
                 { total_time_horizon }, "w");
}

// runRobustRc：运行基于RMPPI控制器仿真（使用鲁棒代价版本）
// 此函数中额外计算了自由能增长等指标
void runRobustRc(const Eigen::Ref<const Eigen::Matrix<float, Dyn::STATE_DIM, total_time_horizon>>& noise)
{
  // 设置初始状态
  DoubleIntegratorDynamics::state_array x;
  x << 2, 0, 0, 1;
  DoubleIntegratorDynamics::state_array xdot;

  // 设置采样器参数
  Sampler::SAMPLING_PARAMS_T sampler_params;
  for (int i = 0; i < Dyn::CONTROL_DIM; i++)
  {
    sampler_params.std_dev[i] = 1;
  }

  // 数据保存容器
  std::vector<float> robust_rc_trajectory(Dyn::STATE_DIM * total_time_horizon, 0);
  std::vector<float> robust_rc_nominal_traj(Dyn::STATE_DIM * num_timesteps * total_time_horizon, 0);
  std::vector<float> robust_rc_nominal_free_energy(total_time_horizon, 0);
  std::vector<float> robust_rc_real_free_energy(total_time_horizon, 0);
  std::vector<float> robust_rc_nominal_free_energy_bound(total_time_horizon, 0);
  std::vector<float> robust_rc_real_free_energy_bound(total_time_horizon, 0);
  std::vector<float> robust_rc_real_free_energy_growth_bound(total_time_horizon, 0);
  std::vector<float> robust_rc_nominal_free_energy_growth(total_time_horizon, 0);
  std::vector<float> robust_rc_real_free_energy_growth(total_time_horizon, 0);
  std::vector<float> robust_rc_nominal_state_used(total_time_horizon, 0);

  // 初始化模型、鲁棒代价函数、采样器及反馈控制器
  Dyn model(100);
  RCost cost;
  auto params = cost.getParams();
  params.crash_cost = 100;
  cost.setParams(params);
  Sampler sampler(sampler_params);

  Feedback fb_controller(&model, dt);
  auto fb_params = fb_controller.getParams();
  fb_params.Q.diagonal() << 500, 500, 100, 100;
  fb_controller.setParams(fb_params);

  // 设置值函数阈值
  float value_function_threshold = 20.0;
  auto controller = RobustMPPIController<Dyn, RCost, Feedback, num_timesteps, 1024, Sampler>(
      &model, &cost, &fb_controller, &sampler, dt, max_iter, lambda, alpha, value_function_threshold);
  auto controller_params = controller.getParams();
  controller_params.dynamics_rollout_dim_ = dim3(64, 1, 1);
  controller_params.cost_rollout_dim_ = dim3(64, 1, 1);
  controller.setParams(controller_params);

  // 开始仿真循环
  for (int t = 0; t < total_time_horizon; ++t)
  {
    /********************** Robust (Robust Cost) **********************/
    controller.updateImportanceSamplingControl(x, 1);
    controller.computeControl(x, 1);
    controller.computeFeedback(x);
    controller.computeFeedbackPropagatedStateSeq();

    auto nominal_trajectory = controller.getTargetStateSeq();
    auto nominal_control = controller.getControlSeq();
    auto fe_stat = controller.getFreeEnergyStatistics();

    // 保存数据以及计算自由能边界和增长指标
    saveState(x, t, robust_rc_trajectory);
    saveTraj(nominal_trajectory, t, robust_rc_nominal_traj);
    robust_rc_nominal_free_energy[t] = fe_stat.nominal_sys.freeEnergyMean;
    robust_rc_real_free_energy[t] = fe_stat.real_sys.freeEnergyMean;
    robust_rc_nominal_free_energy_bound[t] =
        value_function_threshold + 2 * fe_stat.nominal_sys.freeEnergyModifiedVariance;
    robust_rc_real_free_energy_bound[t] = fe_stat.nominal_sys.freeEnergyMean +
                                          cost.getLipshitzConstantCost() * 1 * (x - nominal_trajectory.col(0)).norm();
    robust_rc_real_free_energy_growth_bound[t] = (value_function_threshold - fe_stat.nominal_sys.freeEnergyMean) +
                                                 cost.getLipshitzConstantCost() * 8 * 20 * controller.computeDF() +
                                                 2 * fe_stat.nominal_sys.freeEnergyModifiedVariance;
    robust_rc_nominal_free_energy_growth[t] = fe_stat.nominal_sys.increase;
    robust_rc_real_free_energy_growth[t] = fe_stat.real_sys.increase;
    robust_rc_nominal_state_used[t] = fe_stat.nominal_state_used;

    // 获取并修正控制
    DoubleIntegratorDynamics::control_array current_control = nominal_control.col(0);
    Dyn::control_array fb_control = controller.getFeedbackControl(x, nominal_trajectory.col(0), 0);
    current_control += fb_control;

    // 状态传播和更新
    model.computeDynamics(x, current_control, xdot);
    model.updateState(x, xdot, dt);

    // 添加噪声扰动
    x += noise.col(t) * sqrt(model.getParams().system_noise) * dt;

    // 平移控制序列
    controller.slideControlSequence(1);
  }
  // 保存数据到文件
  cnpy::npy_save("robust_rc_state_trajectory.npy", robust_rc_trajectory.data(),
                 { total_time_horizon, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("robust_rc_nominal_trajectory.npy", robust_rc_nominal_traj.data(),
                 { total_time_horizon, num_timesteps, DoubleIntegratorDynamics::STATE_DIM }, "w");
  cnpy::npy_save("robust_rc_nominal_free_energy.npy", robust_rc_nominal_free_energy.data(), { total_time_horizon },
                 "w");
  cnpy::npy_save("robust_rc_real_free_energy.npy", robust_rc_real_free_energy.data(), { total_time_horizon }, "w");
  cnpy::npy_save("robust_rc_nominal_state_used.npy", robust_rc_nominal_state_used.data(), { total_time_horizon }, "w");
  cnpy::npy_save("robust_rc_real_free_energy_bound.npy", robust_rc_real_free_energy_bound.data(),
                 { total_time_horizon }, "w");
  cnpy::npy_save("robust_rc_nominal_free_energy_bound.npy", robust_rc_nominal_free_energy_bound.data(),
                 { total_time_horizon }, "w");
  cnpy::npy_save("robust_rc_real_free_energy_growth_bound.npy", robust_rc_real_free_energy_growth_bound.data(),
                 { total_time_horizon }, "w");
  cnpy::npy_save("robust_rc_real_free_energy_growth.npy", robust_rc_real_free_energy_growth.data(),
                 { total_time_horizon }, "w");
  cnpy::npy_save("robust_rc_nominal_free_energy_growth.npy", robust_rc_nominal_free_energy_growth.data(),
                 { total_time_horizon }, "w");
}

int main()
{
  // 主函数：运行所有控制器版本的双积分器仿真实验，使用相同的噪声序列，重复20次

  // 创建随机数生成器，用于生成系统噪声（采用正态分布）
  std::mt19937 gen;  // 标准梅森旋转算法引擎
  std::normal_distribution<float> normal_distribution;
  gen.seed(7);  // 使用固定种子7，保证噪声一致性
  normal_distribution = std::normal_distribution<float>(0, 1);

  // 构造一个总时间步长 × 状态维度的噪声矩阵，初始化为零
  Eigen::Matrix<float, Dyn::STATE_DIM, total_time_horizon> universal_noise;
  universal_noise.setZero();

  // 为噪声矩阵生成数据：仅对状态中的第3和第4个分量添加噪声
  for (int t = 0; t < total_time_horizon; ++t)
  {
    for (int i = 2; i < 4; ++i)
    {
      universal_noise(i, t) = normal_distribution(gen);
    }
  }

  // 分别运行各个控制器仿真实验，并输出结束提示
  runVanilla(universal_noise);
  std::cout << "Finished Vanilla" << std::endl;

  runVanillaLarge(universal_noise);
  std::cout << "Finished Vanilla Large" << std::endl;

  runVanillaLargeRC(universal_noise);
  std::cout << "Finished Vanilla Large with Robust Cost" << std::endl;

  runTube(universal_noise);
  std::cout << "Finished Tube with Standard Cost" << std::endl;

  runTubeRC(universal_noise);
  std::cout << "Finished Tube with Robust Cost" << std::endl;

  runRobustSc(universal_noise);
  std::cout << "Finished RMPPI with Standard Cost" << std::endl;

  runRobustRc(universal_noise);
  std::cout << "Finished RMPPI with Robust Cost" << std::endl;

  return 0;
}
