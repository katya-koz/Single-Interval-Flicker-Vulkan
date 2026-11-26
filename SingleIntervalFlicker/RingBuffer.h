#pragma once
#include <vector>
#include <cstddef>

// fixed capacity buffer; oldest element is overwritten when full
template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity) : m_data(capacity) {}

    void push(const T& v) {
        m_data[m_head] = v;
        m_head = (m_head + 1) % m_data.size();
        if (m_size < m_data.size()) ++m_size;
    }

    bool latest(T& out) const {
        if (m_size == 0) return false;
        out = m_data[(m_head + m_data.size() - 1) % m_data.size()];
        return true;
    }

    // most recent `count` elements, oldest -> newest (fewer if the buffer holds less)
    std::vector<T> latest(size_t count) const {
        const size_t n = (count < m_size) ? count : m_size;
        std::vector<T> out;
        out.reserve(n);
        const size_t start = (m_head + m_data.size() - n) % m_data.size();
        for (size_t i = 0; i < n; ++i)
            out.push_back(m_data[(start + i) % m_data.size()]);
        return out;
    }

    // oldest -> newest
    std::vector<T> toVector() const {
        std::vector<T> out;
        out.reserve(m_size);
        const size_t start = (m_head + m_data.size() - m_size) % m_data.size();
        for (size_t i = 0; i < m_size; ++i)
            out.push_back(m_data[(start + i) % m_data.size()]);
        return out;
    }

    void clear() { m_head = 0; m_size = 0; }
    size_t size() const { return m_size; }
    bool empty() const { return m_size == 0; }

private:
    std::vector<T> m_data;
    size_t m_head = 0;
    size_t m_size = 0;
};