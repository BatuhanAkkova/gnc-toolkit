#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <opengnc/utils/quat_utils.hpp>
#include <opengnc/kalman_filters/mekf.hpp>
#include <opengnc/kalman_filters/ukf_attitude.hpp>
#include <opengnc/comm/udp_interface.hpp>

namespace py = pybind11;
using namespace opengnc::utils;
using namespace opengnc::kalman_filters;
using namespace opengnc::comm;

PYBIND11_MODULE(opengnc_py, m) {
    m.doc() = "OpenGNC Python bindings for high-performance C++ filters";

    // --- Quaternion Utilities ---
    m.def("quat_normalize", &quat_normalize, "Normalize a [x,y,z,w] quaternion");
    m.def("quat_mult", &quat_mult, "Multiply two [x,y,z,w] quaternions");
    m.def("quat_conj", &quat_conj, "Congujate of a [x,y,z,w] quaternion");
    m.def("quat_rot", &quat_rot, "Rotate a vector by a [x,y,z,w] quaternion");
    m.def("axis_angle_to_quat", &axis_angle_to_quat, "Convert axis and angle to [x,y,z,w] quaternion");
    m.def("rot_vec_to_quat", &rot_vec_to_quat, "Convert rotation vector (3D) to [x,y,z,w] quaternion");

    // --- MEKF ---
    py::class_<MEKF>(m, "MEKF")
        .def(py::init<const Eigen::Vector4d&, const Eigen::Vector3d&>(),
             py::arg("q_init") = Eigen::Vector4d(0, 0, 0, 1),
             py::arg("beta_init") = Eigen::Vector3d::Zero())
        .def("predict", [](MEKF& self, const Eigen::Vector3d& omega, double dt) {
            self.predict(omega, dt);
        }, py::arg("omega"), py::arg("dt"))
        .def("update", [](MEKF& self, const Eigen::Vector3d& z_body, const Eigen::Vector3d& z_ref) {
            self.update(z_body, z_ref);
        }, py::arg("z_body"), py::arg("z_ref"))
        .def_property_readonly("q", &MEKF::getQuaternion)
        .def_property_readonly("bias", &MEKF::getBias)
        .def_readwrite("P", &MEKF::P)
        .def_readwrite("Q", &MEKF::Q)
        .def_readwrite("R", &MEKF::R);

    // --- UKF Attitude ---
    using UKFAtt = UKF_Attitude<3>;
    py::class_<UKFAtt>(m, "UKF_Attitude")
        .def(py::init<const Eigen::Vector4d&, const Eigen::Vector3d&, double, double, double>(),
             py::arg("q_init") = Eigen::Vector4d(0, 0, 0, 1),
             py::arg("bias_init") = Eigen::Vector3d::Zero(),
             py::arg("alpha") = 1e-2,
             py::arg("beta") = 2.0,
             py::arg("kappa") = 0.0)
        .def("predict", [](UKFAtt& self, double dt, std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)> fx) {
            self.predict(dt, fx);
        }, py::arg("dt"), py::arg("fx"))
        .def("update", [](UKFAtt& self, const Eigen::VectorXd& z, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> hx) {
            self.update(z, hx);
        }, py::arg("z"), py::arg("hx"))
        .def_readwrite("x", &UKFAtt::x)
        .def_readwrite("P", &UKFAtt::P)
        .def_readwrite("Q", &UKFAtt::Q)
        .def_readwrite("R", &UKFAtt::R);

    // --- UDP Communication ---
    py::class_<UDPInterface>(m, "UDPInterface")
        .def(py::init<const std::string&, int, int>())
        .def("open", &UDPInterface::open)
        .def("close", &UDPInterface::close)
        .def("send", &UDPInterface::send)
        .def("receive", &UDPInterface::receive)
        .def_property_readonly("is_open", &UDPInterface::is_open);
}
