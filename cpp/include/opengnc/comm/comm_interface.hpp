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
     */
    virtual bool send(const std::vector<uint8_t>& data) = 0;

    /**
     * Receive data into a buffer.
     * @param buffer Output buffer.
     * @param max_length Maximum bytes to read.
     * @return Number of bytes read.
     */
    virtual int receive(std::vector<uint8_t>& buffer, size_t max_length) = 0;

    /**
     * Check if the interface is open and connected.
     */
    virtual bool is_open() const = 0;
};

} // namespace comm
} // namespace opengnc
