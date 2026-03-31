#pragma once

#include <opengnc/comm/comm_interface.hpp>
#include <memory>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cstring>

namespace opengnc {
namespace comm {

/**
 * CommBridge wraps a CommInterface and provides packet-based framing.
 * Format: [SYNC1 (0x5A)][SYNC2 (0xA5)][ID (1 byte)][LEN (1 byte)][PAYLOAD (0-255)][CRC_H][CRC_L]
 */
class CommBridge {
public:
    static constexpr uint8_t SYNC1 = 0x5A;
    static constexpr uint8_t SYNC2 = 0xA5;

    explicit CommBridge(std::shared_ptr<CommInterface> interface)
        : interface_(std::move(interface)) {}

    /**
     * Send a packet with zero heap allocations.
     * @param msg_id Unique identifier for the message type.
     * @param payload Pointer to the data.
     * @param length Length of the payload (max 240).
     * @return true if successfully sent.
     */
    bool send_packet(uint8_t msg_id, const uint8_t* payload, uint8_t length) {
        if (!interface_ || !interface_->is_open()) return false;
        if (length > 240) return false; // Enforce limit for static buffer

        // Use static/pre-allocated buffer for real-time safety
        // Size = SYNC(2) + ID(1) + LEN(1) + PAYLOAD(max 240) + CRC(2) = 246
        uint8_t buffer[256];
        buffer[0] = SYNC1;
        buffer[1] = SYNC2;
        buffer[2] = msg_id;
        buffer[3] = length;
        
        if (length > 0 && payload) {
            std::memcpy(&buffer[4], payload, length);
        }

        uint16_t crc = calculate_crc(buffer + 2, length + 2);
        buffer[4 + length] = static_cast<uint8_t>((crc >> 8) & 0xFF);
        buffer[5 + length] = static_cast<uint8_t>(crc & 0xFF);

        return interface_->send(buffer, 6 + length);
    }

    /**
     * Parse incoming data and extract packets.
     * @param msg_id [out] ID of the received message.
     * @param payload [out] Extracted payload (must be pre-allocated).
     * @param length [out] Length of the extracted payload.
     * @return true if a complete packet was received and validated.
     */
    bool receive_packet(uint8_t& msg_id, uint8_t* payload, uint8_t& length) {
        if (!interface_ || !interface_->is_open()) return false;

        uint8_t byte;
        while (interface_->receive(&byte, 1) > 0) {
            switch (state_) {
                case State::WAIT_SYNC1:
                    if (byte == SYNC1) state_ = State::WAIT_SYNC2;
                    break;
                case State::WAIT_SYNC2:
                    if (byte == SYNC2) state_ = State::WAIT_ID;
                    else state_ = State::WAIT_SYNC1;
                    break;
                case State::WAIT_ID:
                    temp_id_ = byte;
                    state_ = State::WAIT_LEN;
                    break;
                case State::WAIT_LEN:
                    temp_len_ = byte;
                    receive_count_ = 0;
                    if (temp_len_ > 0) state_ = State::WAIT_PAYLOAD;
                    else state_ = State::WAIT_CRC1;
                    break;
                case State::WAIT_PAYLOAD:
                    temp_payload_[receive_count_++] = byte;
                    if (receive_count_ >= temp_len_) state_ = State::WAIT_CRC1;
                    break;
                case State::WAIT_CRC1:
                    temp_crc_ = static_cast<uint16_t>(byte) << 8;
                    state_ = State::WAIT_CRC2;
                    break;
                case State::WAIT_CRC2:
                    temp_crc_ |= byte;
                    state_ = State::WAIT_SYNC1;

                    // Validate CRC
                    uint8_t head[2] = {temp_id_, temp_len_};
                    uint16_t computed = calculate_crc(head, 2);
                    if (temp_len_ > 0) {
                        computed = calculate_crc_accumulate(computed, temp_payload_, temp_len_);
                    }

                    if (computed == temp_crc_) {
                        msg_id = temp_id_;
                        length = temp_len_;
                        if (length > 0) std::memcpy(payload, temp_payload_, length);
                        return true;
                    }
                    break;
            }
        }
        return false;
    }

    bool is_open() const { return interface_ && interface_->is_open(); }
    void close() { if (interface_) interface_->close(); }

private:
    std::shared_ptr<CommInterface> interface_;

    enum class State {
        WAIT_SYNC1, WAIT_SYNC2, WAIT_ID, WAIT_LEN, WAIT_PAYLOAD, WAIT_CRC1, WAIT_CRC2
    } state_ = State::WAIT_SYNC1;

    uint8_t temp_id_ = 0;
    uint8_t temp_len_ = 0;
    uint8_t temp_payload_[256];
    uint16_t temp_crc_ = 0;
    uint8_t receive_count_ = 0;

    /**
     * Simplified CRC-16 (Fletcher-like or simple sum for HIL speed).
     * Using CRC-16-CCITT for robustness.
     */
    uint16_t calculate_crc(const uint8_t* data, size_t length) const {
        return calculate_crc_accumulate(0xFFFF, data, length);
    }

    uint16_t calculate_crc_accumulate(uint16_t crc, const uint8_t* data, size_t length) const {
        for (size_t i = 0; i < length; ++i) {
            crc ^= (static_cast<uint16_t>(data[i]) << 8);
            for (uint8_t j = 0; j < 8; ++j) {
                if (crc & 0x8000) crc = (crc << 1) ^ 0x1021;
                else crc <<= 1;
            }
        }
        return crc;
    }
};

} // namespace comm
} // namespace opengnc
