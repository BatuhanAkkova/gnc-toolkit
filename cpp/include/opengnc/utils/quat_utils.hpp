#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

namespace opengnc {
namespace utils {

/**
 * OpenGNC uses the [x, y, z, w] convention for quaternions.
 * Eigen::Quaterniond internally stores coefficients as [x, y, z, w],
 * but the (Scalar, Scalar, Scalar, Scalar) constructor expects (w, x, y, z).
 */

/**
 * Standardize 4-element vectors to Eigen::Quaterniond following [x,y,z,w].
 */
inline Eigen::Quaterniond quat_from_array(const Eigen::Vector4d& q) {
    return Eigen::Quaterniond(q(3), q(0), q(1), q(2));
}

inline Eigen::Vector4d quat_to_array(const Eigen::Quaterniond& q) {
    return Eigen::Vector4d(q.x(), q.y(), q.z(), q.w());
}

/**
 * Normalize a quaternion to unit length.
 */
inline Eigen::Vector4d quat_normalize(const Eigen::Vector4d& q) {
    double norm = q.norm();
    if (norm < 1e-15) {
        throw std::runtime_error("Cannot normalize a zero-length quaternion.");
    }
    return q / norm;
}

/**
 * Compute the conjugate of a quaternion.
 */
inline Eigen::Vector4d quat_conj(const Eigen::Vector4d& q) {
    return Eigen::Vector4d(-q(0), -q(1), -q(2), q(3));
}

/**
 * Multiply two quaternions (Hamilton product).
 * Returns q_left * q_right.
 */
inline Eigen::Vector4d quat_mult(const Eigen::Vector4d& q_left, const Eigen::Vector4d& q_right) {
    Eigen::Quaterniond ql = quat_from_array(q_left);
    Eigen::Quaterniond qr = quat_from_array(q_right);
    Eigen::Quaterniond qc = ql * qr;
    return quat_to_array(qc);
}

/**
 * Rotate a 3D vector by a quaternion.
 */
inline Eigen::Vector3d quat_rot(const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
    Eigen::Quaterniond q_eigen = quat_from_array(q);
    return q_eigen * v;
}

/**
 * Create a 3x3 skew-symmetric matrix from a 3D vector.
 */
inline Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0.0,   -v(2),  v(1),
         v(2),   0.0,  -v(0),
        -v(1),   v(0),  0.0;
    return m;
}

/**
 * Convert axis-angle to a unit quaternion [x, y, z, w].
 */
inline Eigen::Vector4d axis_angle_to_quat(const Eigen::Vector3d& axis, double angle) {
    double norm = axis.norm();
    if (norm < 1e-15) {
        return Eigen::Vector4d(0, 0, 0, 1.0);
    }
    Eigen::Vector3d u = axis / norm;
    double s = std::sin(angle / 2.0);
    double c = std::cos(angle / 2.0);
    return Eigen::Vector4d(u(0) * s, u(1) * s, u(2) * s, c);
}

/**
 * Convert a rotation vector (theta_vec) to a unit quaternion [x, y, z, w].
 */
inline Eigen::Vector4d rot_vec_to_quat(const Eigen::Vector3d& rot_vec) {
    double angle = rot_vec.norm();
    return axis_angle_to_quat(rot_vec, angle);
}

} // namespace utils
} // namespace opengnc
