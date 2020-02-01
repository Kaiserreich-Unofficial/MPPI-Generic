#include <gtest/gtest.h>
#include <mppi_core/normexp_kernel_test.cuh>
#include <utils/test_helper.h>
#include <random>
#include <algorithm>
#include <numeric>


class NormExpKernel: public testing::Test {
protected:
    void SetUp() override {
        generator = std::default_random_engine(7.0);
        distribution = std::normal_distribution<float>(100.0,2.0);
    }

    void TearDown() override {

    }

    std::default_random_engine generator;
    std::normal_distribution<float> distribution;
};

TEST_F(NormExpKernel, computeBaselineCost_Test) {
    const int num_rollouts = 4196;
    std::array<float, num_rollouts> cost_vec = {0};

    // Use a range based for loop to set the cost
    for(auto& cost: cost_vec) {
        cost = distribution(generator);
    }

    float min_cost_known = *std::min_element(cost_vec.begin(), cost_vec.end());
    float min_cost_compute = mppi_common::computeBaselineCost(cost_vec.data(), num_rollouts);

    ASSERT_FLOAT_EQ(min_cost_compute, min_cost_known);
}

TEST_F(NormExpKernel, computeNormalizer_Test) {
    const int num_rollouts = 5;
    std::array<float, num_rollouts> cost_vec = {0};

    // Use a range based for loop to set the cost
    for(auto& cost: cost_vec) {
        cost = distribution(generator);
    }

    float sum_cost_known = std::accumulate(cost_vec.begin(), cost_vec.end(), 0.f);
    float sum_cost_compute = mppi_common::computeNormalizer(cost_vec.data(), num_rollouts);

    ASSERT_FLOAT_EQ(sum_cost_compute, sum_cost_known);
}