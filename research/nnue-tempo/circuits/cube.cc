// Evaluate the complete interchange cube for a batch of paired eight-unit activations.
#include <algorithm>
#include <bit>
#include <cstdint>

extern "C" void cube(int rows, const int32_t* pairs, const int32_t* biases,
                     const int32_t* weights, uint8_t* output, uint16_t* continuous) {
    constexpr int inputs = 8, outputs = 32, masks = 256;
    for (int row = 0; row < rows; ++row) {
        const auto* a = pairs + row * 2 * inputs;
        const auto* b = a + inputs;
        int pre[outputs], steps[inputs][outputs];
        for (int j = 0; j < outputs; ++j) {
            pre[j] = biases[j];
            for (int i = 0; i < inputs; ++i) {
                const int w = weights[j * inputs + i];
                pre[j] += a[i] * w;
                steps[i][j] = (b[i] - a[i]) * w;
            }
        }
        for (unsigned n = 0; n < masks; ++n) {
            const unsigned mask = n ^ (n >> 1);
            if (n) {
                const unsigned i = std::countr_zero(n);
                const int direction = ((mask >> i) & 1) ? 1 : -1;
                for (int j = 0; j < outputs; ++j)
                    pre[j] += direction * steps[i][j];
            }
            const int offset = (row * masks + mask) * outputs;
            for (int j = 0; j < outputs; ++j) {
                const int clipped = std::clamp(pre[j], 0, 127 * 64);
                output[offset + j] = static_cast<uint8_t>(clipped >> 6);
                if (continuous) continuous[offset+j] = static_cast<uint16_t>(clipped);
            }
        }
    }
}
