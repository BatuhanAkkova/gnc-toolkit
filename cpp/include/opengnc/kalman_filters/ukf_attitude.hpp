#pragma once

#include <opengnc/kalman_filters/ukf.hpp>
#include <opengnc/utils/quat_utils.hpp>

namespace opengnc {
namespace kalman_filters {

/**
 * Specialized UKF for Attitude Estimation.
 * State (NX=7): [x, y, z, w, bias_x, bias_y, bias_z]
 * Error/Tangent Space (NP=6): [dtheta_x, dtheta_y, dtheta_z, dbias_x, dbias_y, dbias_z]
 * Measurement (NZ): Usually 3 (accelerometer/magnetometer/sun sensor).
 */
template <int NZ = 3>
class UKF_Attitude : public UKF<7, NZ, 6> {
public:
    using Base = UKF<7, NZ, 6>;
    using typename Base::StateVec;
    using typename Base::MeasVec;
    using typename Base::ErrorVec;
    using typename Base::CovMat;
    using typename Base::SigmaPoints;

    UKF_Attitude(const Eigen::Vector4d& q_init = Eigen::Vector4d(0, 0, 0, 1),
                 const Eigen::Vector3d& bias_init = Eigen::Vector3d::Zero(),
                 double alpha = 1e-2, double beta = 2.0, double kappa = 0.0)
        : Base(alpha, beta, kappa) {
        
        // 1. Manifold Subtract: x1 "-" x2 -> dx
        this->subtract_x = [](const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) {
            Eigen::Vector4d q1 = x1.head<4>();
            Eigen::Vector4d q2 = x2.head<4>();
            Eigen::Vector3d b1 = x1.tail<3>();
            Eigen::Vector3d b2 = x2.tail<3>();

            // dq = q2_conj * q1
            Eigen::Vector4d dq = opengnc::utils::quat_mult(opengnc::utils::quat_conj(q2), q1);
            if (dq(3) < 0) dq *= -1.0;

            Eigen::VectorXd dx(6);
            dx.head<3>() = 2.0 * dq.head<3>();
            dx.tail<3>() = b1 - b2;
            return dx;
        };

        // 2. Manifold Add: x "+" dx -> x_new
        this->add_x = [](const Eigen::VectorXd& x, const Eigen::VectorXd& dx) {
            Eigen::Vector4d q = x.head<4>();
            Eigen::Vector3d b = x.tail<3>();
            Eigen::Vector3d dtheta = dx.head<3>();
            Eigen::Vector3d db = dx.tail<3>();

            Eigen::Vector4d dq = opengnc::utils::rot_vec_to_quat(dtheta);
            Eigen::VectorXd x_new(7);
            x_new.head<4>() = opengnc::utils::quat_normalize(opengnc::utils::quat_mult(q, dq));
            x_new.tail<3>() = b + db;
            return x_new;
        };

        // 3. Manifold Mean: weighted average of sigmas
        this->mean_x = [](const Eigen::MatrixXd& sigmas, const Eigen::VectorXd& weights) {
            Eigen::Vector4d q_ref = sigmas.col(0).head<4>();
            Eigen::Vector4d q_avg = Eigen::Vector4d::Zero();
            
            for (int i = 0; i < sigmas.cols(); ++i) {
                Eigen::Vector4d q = sigmas.col(i).head<4>();
                if (q.dot(q_ref) < 0) q *= -1.0;
                q_avg += weights(i) * q;
            }
            
            Eigen::VectorXd x_avg(7);
            x_avg.head<4>() = opengnc::utils::quat_normalize(q_avg);
            x_avg.segment(4, 3) = sigmas.block(4, 0, 3, sigmas.cols()) * weights;
            return x_avg;
        };

        StateVec x_start;
        x_start << q_init, bias_init;
        this->x = x_start;
    }
};

} // namespace kalman_filters
} // namespace opengnc
