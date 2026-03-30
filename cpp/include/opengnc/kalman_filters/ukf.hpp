#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <vector>
#include <functional>

namespace opengnc {
namespace kalman_filters {

/**
 * Generalized Unscented Kalman Filter (UKF).
 * NX: Dimension of the full state vector.
 * NZ: Dimension of the measurement vector.
 * NP: Dimension of the covariance/manifold tangent space (usually NX).
 */
template <int NX, int NZ, int NP = NX>
class UKF {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using StateVec = Eigen::Matrix<double, NX, 1>;
    using MeasVec = Eigen::Matrix<double, NZ, 1>;
    using ErrorVec = Eigen::Matrix<double, NP, 1>;
    using CovMat = Eigen::Matrix<double, NP, NP>;
    using SigmaPoints = Eigen::Matrix<double, NX, 2 * NP + 1>;

    // Manifold operations
    using SubtractFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>;
    using AddFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>;
    using MeanFunc = std::function<Eigen::VectorXd(const Eigen::MatrixXd&, const Eigen::VectorXd&)>;

    // Dynamics and Measurement models
    using FxFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)>;
    using HxFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

    UKF(double alpha = 1e-3, double beta = 2.0, double kappa = 0.0)
        : alpha(alpha), beta(beta), kappa(kappa) {
        
        lambda = alpha * alpha * (NP + kappa) - NP;
        gamma = std::sqrt(NP + lambda);
        
        int num_sigmas = 2 * NP + 1;
        Wm.setZero();
        Wc.setZero();
        
        Wm(0) = lambda / (NP + lambda);
        Wc(0) = Wm(0) + (1.0 - alpha * alpha + beta);
        
        double w = 1.0 / (2.0 * (NP + lambda));
        for (int i = 1; i < num_sigmas; ++i) {
            Wm(i) = w;
            Wc(i) = w;
        }

        // Default vector space operations
        subtract_x = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return (a - b).eval(); };
        add_x = [](const Eigen::VectorXd& x, const Eigen::VectorXd& dx) { return (x + dx).eval(); };
        mean_x = [](const Eigen::MatrixXd& sigs, const Eigen::VectorXd& w) {
            return (sigs * w).eval();
        };

        x.setZero();
        P.setIdentity();
        Q.setIdentity();
        R.setIdentity();
    }

    void predict(double dt, FxFunc fx, const CovMat* Q_custom = nullptr) {
        const CovMat& qm = Q_custom ? *Q_custom : Q;
        int num_sigmas = 2 * NP + 1;

        // 1. Generate sigma points
        SigmaPoints sigmas = generate_sigma_points(x, P);

        // 2. Transform sigma points
        SigmaPoints sigmas_f;
        for (int i = 0; i < num_sigmas; ++i) {
            Eigen::VectorXd s = sigmas.col(i); 
            sigmas_f.col(i) = fx(s, dt).head<NX>();
        }

        // 3. Predicted mean
        this->x = this->mean_x(sigmas_f, this->Wm).head<NX>();

        // 4. Predicted covariance
        this->P.setZero();
        for (int i = 0; i < num_sigmas; ++i) {
            Eigen::VectorXd s_f = sigmas_f.col(i);
            ErrorVec dx = this->subtract_x(s_f, this->x).head<NP>();
            this->P += this->Wc(i) * dx * dx.transpose();
        }
        this->P += qm * dt;
    }

    void update(const MeasVec& z, HxFunc hx, const Eigen::Matrix<double, NZ, NZ>* R_custom = nullptr) {
        const Eigen::Matrix<double, NZ, NZ>& rm = R_custom ? *R_custom : R;
        int num_sigmas = 2 * NP + 1;

        // 1. Regenerate sigma points
        SigmaPoints sigmas_f = generate_sigma_points(this->x, this->P);

        // 2. Transform to measurement space
        Eigen::Matrix<double, NZ, 2 * NP + 1> sigmas_h;
        for (int i = 0; i < num_sigmas; ++i) {
            Eigen::VectorXd s_f = sigmas_f.col(i);
            sigmas_h.col(i) = hx(s_f).head<NZ>();
        }

        // 3. Measurement mean and covariance
        MeasVec zp = sigmas_h * this->Wm;

        Eigen::Matrix<double, NZ, NZ> S;
        S.setZero();
        Eigen::Matrix<double, NP, NZ> Pxz;
        Pxz.setZero();

        for (int i = 0; i < num_sigmas; ++i) {
            Eigen::VectorXd s_h_i = sigmas_h.col(i);
            Eigen::VectorXd dz_dyn = s_h_i - zp;
            MeasVec dz = dz_dyn.head<NZ>();
            
            Eigen::VectorXd s_f = sigmas_f.col(i);
            ErrorVec dx = this->subtract_x(s_f, this->x).head<NP>();

            S += this->Wc(i) * dz * dz.transpose();
            Pxz += this->Wc(i) * dx * dz.transpose();
        }
        S += rm;

        // 4. Correction
        Eigen::Matrix<double, NP, NZ> K = Pxz * S.inverse();
        this->x = this->add_x(this->x, K * (z - zp)).head<NX>();
        this->P = this->P - (K * S * K.transpose());
    }

    SigmaPoints generate_sigma_points(const StateVec& state, const CovMat& cov) {
        SigmaPoints sigmas;
        sigmas.col(0) = state;

        CovMat P_sym = (cov + cov.transpose()) * 0.5;
        // Cholesky decomposition
        Eigen::LLT<CovMat> llt((NP + lambda) * P_sym);
        if (llt.info() != Eigen::Success) {
            // Fallback for non-PSD (numerical issues)
            // In a real flight system, we might want a more robust squareroot
            // but for now we'll just throw or return identity-spread
            throw std::runtime_error("UKF: Covariance matrix not PSD during sigma point generation.");
        }
        CovMat L = llt.matrixL();

        for (int i = 0; i < NP; ++i) {
            Eigen::VectorXd Lcol = L.col(i);
            sigmas.col(i + 1) = add_x(state, Lcol).head<NX>();
            sigmas.col(i + 1 + NP) = add_x(state, -Lcol).head<NX>();
        }
        return sigmas;
    }

    // Parameters and State
    double alpha, beta, kappa, lambda, gamma;
    StateVec x;
    CovMat P, Q;
    Eigen::Matrix<double, NZ, NZ> R;
    Eigen::Matrix<double, 2 * NP + 1, 1> Wm, Wc;

    // Callbacks
    SubtractFunc subtract_x;
    AddFunc add_x;
    MeanFunc mean_x;
};

} // namespace kalman_filters
} // namespace opengnc
