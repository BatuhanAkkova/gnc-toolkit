#include <iostream>
#include <Eigen/Dense>
#include <opengnc/utils/quat_utils.hpp>
#include <opengnc/kalman_filters/mekf.hpp>
#include <opengnc/kalman_filters/ukf_attitude.hpp>
#include <cassert>
#include <cmath>

using namespace opengnc::utils;
using namespace opengnc::kalman_filters;

int main() {
    std::cout << "Starting OpenGNC C++ Test Runner..." << std::endl;
    
    // 1. Eigen Presence
    Eigen::Matrix3d m = Eigen::Matrix3d::Identity();
    std::cout << "Eigen Identity Matrix (3x3):\n" << m << std::endl;
    
    // 2. Quat Utils Verification
    std::cout << "Testing Quat Utils..." << std::endl;
    double pi = std::acos(-1.0);
    
    // Normalization
    Eigen::Vector4d q_unnormalized(1, 1, 1, 1);
    Eigen::Vector4d q_norm = quat_normalize(q_unnormalized);
    assert(std::abs(q_norm.norm() - 1.0) < 1e-12);
    
    // Rotation (90 deg around Z)
    double s45 = std::sin(pi / 4.0);
    double c45 = std::cos(pi / 4.0);
    Eigen::Vector4d q_90z(0, 0, s45, c45);
    Eigen::Vector3d v_x(1, 0, 0);
    Eigen::Vector3d v_rot = quat_rot(q_90z, v_x);
    assert(std::abs(v_rot(0)) < 1e-12);
    assert(std::abs(v_rot(1) - 1.0) < 1e-12);
    
    std::cout << "Quat Utils: PASS" << std::endl;

    // 3. MEKF Verification
    std::cout << "Testing MEKF..." << std::endl;
    MEKF mekf;
    assert(mekf.getQuaternion()(3) == 1.0); // Identity [0,0,0,1]
    
    // Prediction
    Eigen::Vector3d omega(0.1, 0, 0);
    mekf.predict(omega, 1.0);
    assert(mekf.getQuaternion()(0) > 0); 
    
    // Update
    mekf.update(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 1));
    std::cout << "MEKF: PASS" << std::endl;

    // 4. UKF Attitude Verification
    std::cout << "Testing UKF Attitude..." << std::endl;
    UKF_Attitude<3> ukf;
    assert(ukf.x(3) == 1.0); // Identity [0,0,0,1]
    
    // Dynamics Model (Simple 0.1 rad/s X rotation)
    auto dynamics = [](const Eigen::VectorXd& x, double dt) -> Eigen::VectorXd {
        Eigen::Vector4d q = x.head<4>();
        Eigen::Vector3d w(0.1, 0, 0);
        double wm = w.norm();
        if (wm > 1e-10) {
            Eigen::Vector4d dq = axis_angle_to_quat(w/wm, wm*dt);
            q = quat_normalize(quat_mult(q, dq));
        }
        Eigen::VectorXd x_new(7);
        x_new << q, x.tail<3>();
        return x_new;
    };
    
    // Measurement Model (Reference at [0,0,1])
    auto meas_model = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::Vector4d q = x.head<4>();
        Eigen::Vector3d z_ref(0, 0, 1);
        return Eigen::VectorXd(quat_rot(quat_conj(q), z_ref));
    };
    
    // Prediction Step
    ukf.predict(1.0, dynamics);
    assert(ukf.x(0) > 0);
    
    // Update Step
    ukf.update(Eigen::Vector3d(0, 0, 1), meas_model);
    std::cout << "UKF Attitude: PASS" << std::endl;

    std::cout << "All Tests Completed Successfully." << std::endl;
    return 0;
}
