//
// Created by jason on 8/31/22.
//

#include "racer_dubins_elevation.cuh"
#include <mppi/utils/nn_helpers/lstm_lstm_helper.cuh>

#ifndef MPPIGENERIC_RACER_DUBINS_ELEVATION_LSTM_STEERING_CUH
#define MPPIGENERIC_RACER_DUBINS_ELEVATION_LSTM_STEERING_CUH

class RacerDubinsElevationLSTMSteering
  : public RacerDubinsElevationImpl<RacerDubinsElevationLSTMSteering, RacerDubinsElevationParams>
{
public:
  using PARENT_CLASS = RacerDubinsElevationImpl<RacerDubinsElevationLSTMSteering, RacerDubinsElevationParams>;

  RacerDubinsElevationLSTMSteering(int init_input_dim, int init_hidden_dim, std::vector<int>& init_output_layers,
                                   int input_dim, int hidden_dim, std::vector<int>& output_layers, int init_len,
                                   cudaStream_t stream = nullptr);
  RacerDubinsElevationLSTMSteering(std::string path, cudaStream_t stream = nullptr);

  std::string getDynamicsModelName() const override
  {
    return "RACER Dubins LSTM Steering Model";
  }

  void bindToStream(cudaStream_t stream);

  void GPUSetup();

  void freeCudaMem();

  void updateFromBuffer(const buffer_trajectory& buffer);

  void step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state, Eigen::Ref<state_array> state_der,
            const Eigen::Ref<const control_array>& control, Eigen::Ref<output_array> output, const float t,
            const float dt);

  void initializeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                          Eigen::Ref<output_array> output, float t_0, float dt);

  __device__ inline void step(float* state, float* next_state, float* state_der, float* control, float* output,
                              float* theta_s, const float t, const float dt);

  __device__ void initializeDynamics(float* state, float* control, float* output, float* theta_s, float t_0, float dt);

  __device__ void updateState(float* state, float* next_state, float* state_der, const float dt);

  void updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
                   Eigen::Ref<state_array> state_der, const float dt);

  std::shared_ptr<LSTMLSTMHelper<>> getHelper()
  {
    return lstm_lstm_helper_;
  }

  LSTMHelper<>* network_d_ = nullptr;

protected:
  std::shared_ptr<LSTMLSTMHelper<>> lstm_lstm_helper_;
};

#if __CUDACC__
#include "racer_dubins_elevation_lstm_steering.cu"
#endif

#endif  // MPPIGENERIC_RACER_DUBINS_ELEVATION_LSTM_STEERING_CUH
