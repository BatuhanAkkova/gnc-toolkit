#pragma once

#include <opengnc/comm/comm_interface.hpp>
#include <asio.hpp>
#include <iostream>

namespace opengnc {
namespace comm {

/**
 * UDPInterface implements CommInterface using Asio.
 * High performance UDP telemetry/command transport.
 */
class UDPInterface : public CommInterface {
public:
    /**
     * @param remote_ip External target address.
     * @param remote_port External target port.
     * @param local_port Port to bind for incoming data.
     */
    UDPInterface(const std::string& remote_ip, int remote_port, int local_port)
        : remote_ip_(remote_ip), remote_port_(remote_port), local_port_(local_port),
          io_context_(), socket_(io_context_) {}

    ~UDPInterface() {
        close();
    }

    bool open() override {
        try {
            // Local Binding
            socket_.open(asio::ip::udp::v4());
            socket_.bind(asio::ip::udp::endpoint(asio::ip::udp::v4(), local_port_));
            socket_.non_blocking(true);

            // Remote Endpoint
            remote_endpoint_ = asio::ip::udp::endpoint(
                asio::ip::address::from_string(remote_ip_), remote_port_
            );

            is_open_ = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[UDPInterface] Open Error: " << e.what() << std::endl;
            return false;
        }
    }

    void close() override {
        if (socket_.is_open()) {
            socket_.close();
        }
        is_open_ = false;
    }

    bool send(const std::vector<uint8_t>& data) override {
        if (!is_open_) return false;
        try {
            socket_.send_to(asio::buffer(data), remote_endpoint_);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[UDPInterface] Send Error: " << e.what() << std::endl;
            return false;
        }
    }

    int receive(std::vector<uint8_t>& buffer, size_t max_length) override {
        if (!is_open_) return 0;
        try {
            buffer.resize(max_length);
            asio::ip::udp::endpoint sender_endpoint;
            size_t len = socket_.receive_from(asio::buffer(buffer), sender_endpoint);
            buffer.resize(len);
            return static_cast<int>(len);
        } catch (const asio::system_error& e) {
            // asio::error::would_block is common in non-blocking mode
            if (e.code() == asio::error::would_block) {
                return 0;
            }
            std::cerr << "[UDPInterface] Receive Error: " << e.what() << std::endl;
            return -1;
        }
    }

    bool is_open() const override {
        return is_open_;
    }

private:
    std::string remote_ip_;
    int remote_port_;
    int local_port_;

    asio::io_context io_context_;
    asio::ip::udp::socket socket_;
    asio::ip::udp::endpoint remote_endpoint_;
    
    bool is_open_ = false;
};

} // namespace comm
} // namespace opengnc
