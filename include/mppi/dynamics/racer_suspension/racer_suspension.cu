#include <mppi/dynamics/racer_suspension/racer_suspension.cuh>
#include <mppi/utils/eigen_type_conversions.h>
#include <mppi/utils/math_utils.h>

void RacerSuspension::GPUSetup()
{
  auto* derived = static_cast<Dynamics<RacerSuspension, RacerSuspensionParams, 15, 2>*>(this);
  tex_helper_->GPUSetup();
  derived->GPUSetup();
}

void RacerSuspension::freeCudaMem()
{
  tex_helper_->freeCudaMem();
}

void RacerSuspension::paramsToDevice()
{
  if (this->GPUMemStatus_)
  {
    // does all the internal texture updates
    tex_helper_->copyToDevice();
    // makes sure that the device ptr sees the correct texture object
    HANDLE_ERROR(cudaMemcpyAsync(&(this->model_d_->tex_helper_), &(tex_helper_->ptr_d_),
                                 sizeof(TwoDTextureHelper<float>*), cudaMemcpyHostToDevice, this->stream_));
  }
  Dynamics<RacerSuspension, RacerSuspensionParams, 15, 2>::paramsToDevice();
}

// combined computeDynamics & updateState
void RacerSuspension::step(Eigen::Ref<state_array> state, Eigen::Ref<const control_array> control, const float dt)
{
  state_array x_dot;
  Eigen::Matrix3f omegaJac;
  computeDynamics(state, control, x_dot, &omegaJac);
  // approximate implicit euler for angular rate states
  Eigen::Vector3f deltaOmega =
      (Eigen::Matrix3f::Identity() - dt * omegaJac).inverse() * dt * x_dot.segment(STATE_OMEGA, 3);
  state_array delta_x = x_dot * dt;
  delta_x.segment(STATE_OMEGA, 3) = deltaOmega;
  state += delta_x;
  float q_norm = state.segment(STATE_Q, 4).norm();
  state.segment(STATE_Q, 4) /= q_norm;
}

void RacerSuspension::updateState(Eigen::Ref<state_array> state, Eigen::Ref<state_array> state_der, const float dt)
{
  state += state_der * dt;
  float q_norm = state.segment(STATE_Q, 4).norm();
  state.segment(STATE_Q, 4) /= q_norm;
  state_der.setZero();
}

__device__ void RacerSuspension::updateState(float* state, float* state_der, const float dt)
{
  unsigned int i;
  unsigned int tdy = threadIdx.y;
  // Add the state derivative time dt to the current state.
  // printf("updateState thread %d, %d = %f, %f\n", threadIdx.x, threadIdx.y, state[0], state_der[0]);
  for (i = tdy; i < STATE_DIM; i += blockDim.y)
  {
    state[i] += state_der[i] * dt;
  }
  // TODO renormalize quaternion
}

static float stribeck_friction(float v, float mu_s, float v_slip, float& partial_mu_partial_v)
{
    float mu = v / v_slip * mu_s;
    partial_mu_partial_v = 0;
    if (mu > mu_s)
    {
      return mu_s;
    }
    if (mu < -mu_s)
    {
      return -mu_s;
    }
    partial_mu_partial_v = mu_s / v_slip;
    return mu;
}

void RacerSuspension::computeDynamics(const Eigen::Ref<const state_array>& state,
                                      const Eigen::Ref<const control_array>& control, Eigen::Ref<state_array> state_der,
                                      Eigen::Matrix3f* omegaJacobian)
{
  Eigen::Vector3f p_I = state.segment(STATE_P, 3);
  Eigen::Vector3f v_I = state.segment(STATE_V, 3);
  Eigen::Vector3f omega = state.segment(STATE_OMEGA, 3);
  Eigen::Quaternionf q(state[STATE_QW], state[STATE_QX], state[STATE_QY], state[STATE_QZ]);
  Eigen::Matrix3f R = q.toRotationMatrix();
  float tan_delta = tan(state[STATE_STEER]);
  Eigen::Matrix3f Rdot = R * mppi_math::skewSymmetricMatrix(omega);

  // linear engine model
  float vel_x = (R.transpose() * v_I)[0]; 
  float throttle = max(0.0, control[CTRL_THROTTLE_BRAKE]);
  float brake = max(0.0, -control[CTRL_THROTTLE_BRAKE]);
  float acc = params_.c_t * throttle - copysign(params_.c_b * brake, vel_x) - params_.c_v * vel_x +  params_.c_0;
  float propulsion_force = params_.mass * acc;

  Eigen::Vector3f f_B = Eigen::Vector3f::Zero();
  Eigen::Vector3f tau_B = Eigen::Vector3f::Zero();
  Eigen::Matrix3f tau_B_jac = Eigen::Matrix3f::Zero();
  Eigen::Vector3f f_r_B[4];
  // for each wheel
  for (int i=0; i < 4; i++) {
    // compute suspension from elevation map
    Eigen::Vector3f p_w_nom_B_i = cudaToEigen(params_.wheel_pos_wrt_base_link[i]) - cudaToEigen(params_.cg_pos_wrt_base_link);
    Eigen::Vector3f p_w_nom_I_i = p_I + R * p_w_nom_B_i;
    float h_i = 0;
    Eigen::Vector3f n_I_i(0, 0, 1);
    if (tex_helper_->checkTextureUse(TEXTURE_ELEVATION_MAP))
    {
      float4 wheel_elev = tex_helper_->queryTextureAtWorldPose(TEXTURE_ELEVATION_MAP, EigenToCuda(p_w_nom_I_i));
      h_i = wheel_elev.w;
      // std::cout << "h_i " << h_i << std::endl;
      n_I_i = Eigen::Vector3f(wheel_elev.x, wheel_elev.y, wheel_elev.z);
    }
    Eigen::Vector3f p_c_I_i(p_w_nom_I_i[0], p_w_nom_I_i[1], h_i);
    float l_i = p_w_nom_I_i[2] - p_c_I_i[2];
    Eigen::Vector3f p_dot_w_nom_I_i = v_I + Rdot * p_w_nom_B_i;
    Eigen::Matrix3f p_dot_w_nom_I_i_Jac = R * Eigen::Matrix3f::Identity().colwise().cross(p_w_nom_B_i);
    float h_dot_i = -n_I_i[0] / n_I_i[2] * p_dot_w_nom_I_i[0] - n_I_i[1] / n_I_i[2] * p_dot_w_nom_I_i[1];
    Eigen::RowVector3f h_dot_i_Jac =
        -n_I_i[0] / n_I_i[2] * p_dot_w_nom_I_i_Jac.row(0) - n_I_i[1] / n_I_i[2] * p_dot_w_nom_I_i_Jac.row(1);
    float l_dot_i = p_dot_w_nom_I_i[2] - h_dot_i;
    Eigen::RowVector3f l_dot_i_Jac = p_dot_w_nom_I_i_Jac.row(2) - h_dot_i_Jac;

    float f_k_i = -params_.k_s[i] * (l_i - params_.l_0[i]) - params_.c_s[i] * l_dot_i;
    Eigen::RowVector3f f_k_i_Jac = -params_.c_s[i] * l_dot_i_Jac;
    if (f_k_i < 0)
    {
      f_k_i = 0;
      f_k_i_Jac = Eigen::RowVector3f::Zero();
    }

    // contact frame
    Eigen::Vector3f p_dot_c_I_i(p_dot_w_nom_I_i[0], p_dot_w_nom_I_i[1], h_dot_i);
    Eigen::Matrix3f p_dot_c_I_i_Jac;
    p_dot_c_I_i_Jac.row(0) = p_dot_w_nom_I_i_Jac.row(0);
    p_dot_c_I_i_Jac.row(1) = p_dot_w_nom_I_i_Jac.row(1);
    p_dot_c_I_i_Jac.row(2) = h_dot_i_Jac;
    float delta_i;
    if (i == RacerSuspensionParams::WHEEL_FRONT_LEFT) {
      delta_i = atan( params_.wheel_base * tan_delta / (params_.wheel_base - tan_delta * params_.width / 2));
    } else if (i == RacerSuspensionParams::WHEEL_FRONT_RIGHT) {
      delta_i = atan( params_.wheel_base * tan_delta / (params_.wheel_base + tan_delta * params_.width / 2));
    } else { // rear wheels
      delta_i = 0;
    }

    Eigen::Vector3f n_B_i = R.transpose() * n_I_i;
    Eigen::Vector3f wheel_dir_B_i(cos(delta_i), sin(delta_i), 0);
    Eigen::Vector3f s_B_i = n_B_i.cross(wheel_dir_B_i).normalized();
    Eigen::Vector3f t_B_i = s_B_i.cross(n_B_i);
    Eigen::Matrix3f R_C_i_to_B;
    R_C_i_to_B.col(0) = t_B_i;
    R_C_i_to_B.col(1) = s_B_i;
    R_C_i_to_B.col(2) = n_B_i;

    // contact velocity
    Eigen::Vector3f p_dot_c_B_i = R.transpose() * p_dot_c_I_i;
    Eigen::Matrix3f p_dot_c_B_i_Jac = R.transpose() * p_dot_c_I_i_Jac;
    float v_w_t_i = t_B_i.dot(p_dot_c_B_i);
    float v_w_s_i = s_B_i.dot(p_dot_c_B_i);
    Eigen::RowVector3f v_w_s_i_Jac = s_B_i.transpose() * p_dot_c_B_i_Jac;

    // compute contact forces
    float f_n_i = f_k_i;
    float partial_mu_s_partial_v;
    float mu_s = stribeck_friction(v_w_s_i, params_.mu, params_.v_slip, partial_mu_s_partial_v);
    float f_s_i = -mu_s * f_n_i;
    float f_t_i = max(-params_.mu * f_n_i, min(propulsion_force / 4, params_.mu * f_n_i));
    Eigen::RowVector3f f_n_i_Jac = f_k_i_Jac;
    Eigen::RowVector3f f_t_i_Jac = Eigen::RowVector3f::Zero();
    if (propulsion_force / 4 > params_.mu * f_n_i)
    {
      f_t_i_Jac = params_.mu * f_n_i_Jac;
    }
    else if (propulsion_force / 4 < -params_.mu * f_n_i)
    {
      f_t_i_Jac = -params_.mu * f_n_i_Jac;
    }
    Eigen::RowVector3f f_s_i_Jac = -f_n_i * partial_mu_s_partial_v * v_w_s_i_Jac - mu_s * f_n_i_Jac;

    // contact force & location
    Eigen::Vector3f f_r_C_i(f_t_i, f_s_i, f_n_i);
    Eigen::Matrix3f f_r_C_i_Jac;
    f_r_C_i_Jac.row(0) = f_t_i_Jac;
    f_r_C_i_Jac.row(1) = f_s_i_Jac;
    f_r_C_i_Jac.row(2) = f_n_i_Jac;
    f_r_B[i] = R_C_i_to_B * f_r_C_i;
    Eigen::Matrix3f f_r_B_i_Jac = R_C_i_to_B = f_r_C_i_Jac;
    Eigen::Vector3f p_c_B_i = R.transpose() * (p_c_I_i - p_I);

    // accumulate forces & moments
    f_B += f_r_B[i];
    tau_B += p_c_B_i.cross(f_r_B[i]);
    tau_B_jac += -f_r_B_i_Jac.colwise().cross(p_c_B_i);
  }

  Eigen::Vector3f g(0, 0, -params_.gravity);

  state_der.segment(STATE_P, 3) = v_I;
  state_der.segment(STATE_V, 3) = 1 / params_.mass * R * f_B + g;
  Eigen::Quaternionf qdot;
  qdot.coeffs() = 0.5 * (q * Eigen::Quaternionf(0, omega[0], omega[1], omega[2])).coeffs();
  state_der[STATE_QW] = qdot.w();
  state_der[STATE_QX] = qdot.x();
  state_der[STATE_QY] = qdot.y();
  state_der[STATE_QZ] = qdot.z();
  Eigen::Vector3f J_diag(params_.Jxx, params_.Jyy, params_.Jzz);
  Eigen::Vector3f J_inv_diag(1/params_.Jxx, 1/params_.Jyy, 1/params_.Jzz);
  state_der.segment(STATE_OMEGA, 3) = J_inv_diag.cwiseProduct(J_diag.cwiseProduct(omega).cross(omega) + tau_B);
  if (omegaJacobian)
  {
    Eigen::Matrix3f J = J_diag.asDiagonal();
    Eigen::Matrix3f Jwxw_jac =
        J.colwise().cross(omega) - Eigen::Matrix3f::Identity().colwise().cross(J_diag.cwiseProduct(omega));
    *omegaJacobian = J_inv_diag.asDiagonal() * (Jwxw_jac + tau_B_jac);
  }

  // Steering actuator model
  float steer = control[CTRL_STEER_CMD] / params_.steer_command_angle_scale;
  state_der[STATE_STEER] = params_.steering_constant * (steer - state[STATE_STEER]);
  state_der[STATE_STEER_VEL] = 0; // Not used in first order model
}

__device__ void RacerSuspension::computeDynamics(float* state, float* control, float* state_der, float* theta_s)
{

}

Eigen::Quaternionf RacerSuspension::get_attitude(const Eigen::Ref<const state_array>& state)
{
  return Eigen::Quaternionf(state[STATE_QW], state[STATE_QX], state[STATE_QY], state[STATE_QZ]);
}

Eigen::Vector3f RacerSuspension::get_position(const Eigen::Ref<const state_array>& state)
{
  Eigen::Vector3f p_COM(state[STATE_PX], state[STATE_PY], state[STATE_PZ]);
  Eigen::Quaternionf q = get_attitude(state);
  return p_COM - q * cudaToEigen(params_.cg_pos_wrt_base_link);
}
