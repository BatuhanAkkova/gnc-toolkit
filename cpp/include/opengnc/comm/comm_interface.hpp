#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace opengnc {
namespace comm {

/**
 * CommInterface Defines the base abstraction for HIL/SIL communication.
 */
class CommInterface {
public:
    virtual ~CommInterface() = default;

    /**
     * Open the communication channel.
     * @return true if successful.
     */
    virtual bool open() = 0;

    /**
     * Close the communication channel.
     */
    virtual void close() = 0;

    /**
     * Send a buffer of data.
     * @param data Pointer to the data.
     * @param length Number of bytes to send.
     * @return true if successful.
     */
    virtual bool send(const uint8_t* data, size_t length) = 0;

    /**
     * Send a buffer of data (convenience overload).
     */
    virtual bool send(const std::vector<uint8_t>& data) {
        return send(data.data(), data.size());
    }

    /**
     * Receive data into a buffer.
     * @param buffer Output buffer pointer.
     * @param max_length Maximum bytes to read.
     * @return Number of bytes read, or negative on error.
     */
    virtual int receive(uint8_t* buffer, size_t max_length) = 0;

    /**
     * Receive data into a vector (convenience overload).
     * Note: This may involve buffer resizing.
     */
    virtual int receive(std::vector<uint8_t>& buffer, size_t max_length) {
        if (buffer.size() < max_length) {
            buffer.resize(max_length);
        }
        int len = receive(buffer.data(), max_length);
        if (len > 0) {
            buffer.resize(len);
        }
        return len;
    }

    /**
     * Check if the interface is open and connected.
     */
    virtual bool is_open() const = 0;

    /**
     * Return number of bytes available to read without blocking.
     */
    virtual size_t available() const { return 0; }
};

} // namespace comm
} // namespace opengnc
