#pragma once

#include <opengnc/comm/comm_interface.hpp>
#include <asio.hpp>
#include <iostream>

namespace opengnc {
namespace comm {

/**
 * SerialInterface implements CommInterface using Asio serial_port.
 * Used for HIL with microcontrollers (Teensy, STM32, ESP32).
 */
class SerialInterface : public CommInterface {
public:
    using CommInterface::send;
    using CommInterface::receive;

    /**
     * @param port_name Name of the serial port (e.g., "COM3" or "/dev/ttyUSB0").
     * @param baud_rate Speed of the connection (e.g., 115200, 921600).
     */
    SerialInterface(const std::string& port_name, unsigned int baud_rate = 115200)
        : port_name_(port_name), baud_rate_(baud_rate),
          io_context_(), port_(io_context_) {}

    ~SerialInterface() {
        close();
    }

    bool open() override {
        try {
            port_.open(port_name_);
            port_.set_option(asio::serial_port_base::baud_rate(baud_rate_));
            port_.set_option(asio::serial_port_base::character_size(8));
            port_.set_option(asio::serial_port_base::stop_bits(asio::serial_port_base::stop_bits::one));
            port_.set_option(asio::serial_port_base::parity(asio::serial_port_base::parity::none));
            port_.set_option(asio::serial_port_base::flow_control(asio::serial_port_base::flow_control::none));
            
            is_open_ = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[SerialInterface] Open Error (" << port_name_ << "): " << e.what() << std::endl;
            return false;
        }
    }

    void close() override {
        if (port_.is_open()) {
            port_.close();
        }
        is_open_ = false;
    }

    bool send(const uint8_t* data, size_t length) override {
        if (!is_open_) return false;
        try {
            asio::write(port_, asio::buffer(data, length));
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[SerialInterface] Send Error: " << e.what() << std::endl;
            return false;
        }
    }

    int receive(uint8_t* buffer, size_t max_length) override {
        if (!is_open_) return 0;
        try {
            // Non-blocking read or partial read
            // In HIL, we often want to read whatever is available
            size_t len = port_.read_some(asio::buffer(buffer, max_length));
            return static_cast<int>(len);
        } catch (const asio::system_error& e) {
            if (e.code() == asio::error::would_block) {
                return 0;
            }
            std::cerr << "[SerialInterface] Receive Error: " << e.what() << std::endl;
            return -1;
        }
    }

    bool is_open() const override {
        return is_open_;
    }

    /**
     * Set a new baud rate on a closed port.
     */
    void set_baud_rate(unsigned int baud) {
        baud_rate_ = baud;
        if (is_open_) {
            port_.set_option(asio::serial_port_base::baud_rate(baud_rate_));
        }
    }

private:
    std::string port_name_;
    unsigned int baud_rate_;

    asio::io_context io_context_;
    asio::serial_port port_;
    
    bool is_open_ = false;
};

} // namespace comm
} // namespace opengnc
