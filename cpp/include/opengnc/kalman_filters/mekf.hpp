#pragma once

#include <Eigen/Dense>
#include <opengnc/utils/quat_utils.hpp>

namespace opengnc {
namespace kalman_filters {

/**
 * Multiplicative Extended Kalman Filter (MEKF) for Attitude Estimation.
 */
class MEKF {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @param q_init Initial [x, y, z, w] quaternion.
     * @param beta_init Initial gyro bias [bx, by, bz].
     */
    MEKF(const Eigen::Vector4d& q_init = Eigen::Vector4d(0, 0, 0, 1),
         const Eigen::Vector3d& beta_init = Eigen::Vector3d::Zero())
        : q(opengnc::utils::quat_normalize(q_init)), beta(beta_init) {
        
        P.setIdentity();
        P *= 0.1;
        
        Q.setIdentity();
        Q *= 0.001;
        
        R.setIdentity();
        R *= 0.01;
    }

    /**
     * Prediction step.
     * @param omega_meas Measured angular rate (rad/s).
     * @param dt Time step (s).
     * @param Q_custom Optional override for process noise.
     */
    void predict(const Eigen::Vector3d& omega_meas, double dt, 
                 const Eigen::Matrix<double, 6, 6>* Q_custom = nullptr) {
        
        const Eigen::Matrix<double, 6, 6>& qm = Q_custom ? *Q_custom : Q;
        
        // 1. Integrate Reference State
        Eigen::Vector3d omega = omega_meas - beta;
        double wm = omega.norm();
        
        if (wm > 1e-10) {
            Eigen::Vector3d axis = omega / wm;
            double angle = wm * dt;
            Eigen::Vector4d dq = opengnc::utils::axis_angle_to_quat(axis, angle);
            q = opengnc::utils::quat_mult(q, dq);
        }
        q = opengnc::utils::quat_normalize(q);
        
        // 2. Propagate Error Covariance
        Eigen::Matrix3d wx = opengnc::utils::skew_symmetric(omega);
        Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Zero();
        F.block<3, 3>(0, 0) = -wx;
        F.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
        
        Eigen::Matrix<double, 6, 6> Phi = Eigen::Matrix<double, 6, 6>::Identity() + F * dt;
        P = (Phi * P * Phi.transpose()) + (qm * dt);
    }

    /**
     * Update step using a vector measurement (e.g. Magnetometer, Sun Sensor).
     * @param z_body Measured vector in body frame.
     * @param z_ref Reference vector in inertial frame.
     * @param R_custom Optional override for measurement noise.
     */
    void update(const Eigen::Vector3d& z_body, const Eigen::Vector3d& z_ref,
                const Eigen::Matrix3d* R_custom = nullptr) {
        
        const Eigen::Matrix3d& r = R_custom ? *R_custom : R;
        
        // 1. Predicted measurement: zp = C(q) * z_ref
        Eigen::Vector3d zp = opengnc::utils::quat_rot(opengnc::utils::quat_conj(q), z_ref);
        
        // 2. Sensitivity matrix H = [ [zp]x | 0_{3x3} ]
        Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
        H.block<3, 3>(0, 0) = opengnc::utils::skew_symmetric(zp);
        
        // 3. Kalman Gain
        Eigen::Matrix3d S = H * P * H.transpose() + r;
        Eigen::Matrix<double, 6, 3> K = P * H.transpose() * S.inverse();
        
        // 4. Correct error state
        Eigen::Matrix<double, 6, 1> dx = K * (z_body - zp);
        Eigen::Vector3d dtheta = dx.segment<3>(0);
        Eigen::Vector3d dbeta = dx.segment<3>(3);
        
        // 5. Apply corrections
        // Multiplicative correction: q = q * dq(dtheta/2)
        Eigen::Vector4d dq_corr;
        dq_corr << 0.5 * dtheta, 1.0; 
        q = opengnc::utils::quat_normalize(opengnc::utils::quat_mult(q, dq_corr));
        beta += dbeta;
        
        // 6. Covariance Update (Joseph Form for stability)
        Eigen::Matrix<double, 6, 6> I_KH = Eigen::Matrix<double, 6, 6>::Identity() - K * H;
        P = I_KH * P * I_KH.transpose() + K * r * K.transpose();
    }

    // Accessors
    Eigen::Vector4d getQuaternion() const { return q; }
    Eigen::Vector3d getBias() const { return beta; }
    Eigen::Matrix<double, 6, 6> getCovariance() const { return P; }

    // Mutators for tuning
    void setProcessNoise(const Eigen::Matrix<double, 6, 6>& Q_new) { Q = Q_new; }
    void setMeasurementNoise(const Eigen::Matrix3d& R_new) { R = R_new; }

    Eigen::Vector4d q;     // [x, y, z, w]
    Eigen::Vector3d beta;  // Gyro bias
    Eigen::Matrix<double, 6, 6> P; // Covariance
    Eigen::Matrix<double, 6, 6> Q; // Process Noise
    Eigen::Matrix3d R;             // Measurement Noise
};

} // namespace kalman_filters
} // namespace opengnc
