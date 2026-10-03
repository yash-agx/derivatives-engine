// Monte Carlo option pricing core in C++17.
//
// This is the same four estimators as src/pricing/variance_reduction.py, built
// to be called from Python through a plain C ABI (see mc_run at the bottom and
// src/pricing/cpp_backend.py). The point is not only speed: running them
// through the Phase 3 harness is what shows they are statistically correct --
// unbiased, with an honest standard error -- and not merely fast.
//
// Two design decisions carry most of the weight.
//
// 1. COUNTER-BASED RANDOM NUMBERS.
//    The generator is a pure function of (seed, index): path i always gets the
//    same normal draws no matter which thread computes it, or how many threads
//    there are. That is what guarantees two things at once -- threads cannot
//    overlap (each index is handed out exactly once) and the answer does not
//    depend on the thread count. A conventional sequential generator split
//    across threads gives neither guarantee for free.
//
// 2. A FIXED NUMBER OF CHUNKS, INDEPENDENT OF THREAD COUNT.
//    Floating-point addition is not associative, so if threads accumulated
//    their own partial sums the answer would shift in the last bits when the
//    thread count changed. Instead the work is always divided into the same
//    CHUNKS whatever the thread count; threads take whole chunks, and the
//    chunk results are combined in chunk order at the end. The reduction tree
//    is therefore fixed, and results are bitwise identical on 1 thread and on
//    16.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

// ---------------------------------------------------------------------------
// Counter-based random numbers
// ---------------------------------------------------------------------------

// SplitMix64's finalizer: a bijective mix strong enough that stepping a counter
// through it is itself a well-tested generator. Being a pure function is the
// property we need -- there is no state to carry between threads.
inline std::uint64_t mix64(std::uint64_t x) {
    x += kGolden;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline std::uint64_t raw_at(std::uint64_t seed, std::uint64_t counter) {
    // Mixing the seed in separately keeps nearby seeds from producing nearby
    // streams, which matters because the experiments use seeds 0, 1, 2, ...
    return mix64(mix64(seed) ^ (counter * kGolden));
}

// A uniform strictly inside (0, 1): the 0 would make log(u) infinite below.
inline double uniform_at(std::uint64_t seed, std::uint64_t counter) {
    return (static_cast<double>(raw_at(seed, counter) >> 11) + 0.5) *
           (1.0 / 9007199254740992.0);  // 2^-53
}

// Box-Muller: two uniforms in, two independent standard normals out. Indexing
// by block (rather than by draw) keeps both halves of a transform together, so
// nothing is computed twice and the mapping stays a pure function of the block.
inline void normals_at(std::uint64_t seed, std::uint64_t block, double& z0,
                       double& z1) {
    const double u1 = uniform_at(seed, 2 * block);
    const double u2 = uniform_at(seed, 2 * block + 1);
    const double radius = std::sqrt(-2.0 * std::log(u1));
    z0 = radius * std::cos(2.0 * kPi * u2);
    z1 = radius * std::sin(2.0 * kPi * u2);
}

// ---------------------------------------------------------------------------
// Payoffs
// ---------------------------------------------------------------------------

enum Kind { kCall = 0, kPut = 1, kStraddle = 2 };
enum Method { kPlain = 0, kAntithetic = 1, kControl = 2, kImportance = 3 };

inline double payoff_of(double spot, double strike, int kind) {
    switch (kind) {
        case kCall: return spot > strike ? spot - strike : 0.0;
        case kPut: return strike > spot ? strike - spot : 0.0;
        default: return std::fabs(spot - strike);
    }
}

// ---------------------------------------------------------------------------
// Welford accumulation, in a form that combines across chunks
// ---------------------------------------------------------------------------
//
// Means and variances are accumulated in one pass without ever forming a sum of
// squares, which would lose most of its precision to cancellation at 10^7
// paths. The control variate also needs a covariance, so the accumulator
// carries a second variable and their co-moment.

struct Moments {
    double count = 0.0;
    double mean_y = 0.0, m2_y = 0.0;
    double mean_x = 0.0, m2_x = 0.0, c2_xy = 0.0;

    inline void add(double y, double x) {
        count += 1.0;
        const double dy = y - mean_y;
        const double dx = x - mean_x;
        mean_y += dy / count;
        mean_x += dx / count;
        // Welford: the second term uses the UPDATED mean, giving the stable
        // form. The co-moment mixes one updated and one previous delta.
        m2_y += dy * (y - mean_y);
        m2_x += dx * (x - mean_x);
        c2_xy += dx * (y - mean_y);
    }

    inline void merge(const Moments& other) {
        if (other.count == 0.0) return;
        if (count == 0.0) { *this = other; return; }
        const double total = count + other.count;
        const double dy = other.mean_y - mean_y;
        const double dx = other.mean_x - mean_x;
        const double scale = count * other.count / total;
        m2_y += other.m2_y + dy * dy * scale;
        m2_x += other.m2_x + dx * dx * scale;
        c2_xy += other.c2_xy + dx * dy * scale;
        mean_y += dy * other.count / total;
        mean_x += dx * other.count / total;
        count = total;
    }
};

struct Problem {
    double S, K, r, sigma, T;
    int kind;
    int method;
    double drift, vol, discount, shift;
};

// One chunk of work: blocks [first, last). A block is one Box-Muller transform,
// so it carries two normal draws and therefore two paths (or two antithetic
// pairs). Everything here depends only on the block index, never on which
// thread runs it.
Moments run_chunk(const Problem& p, std::uint64_t seed, std::uint64_t first,
                  std::uint64_t last, std::uint64_t unit_limit) {
    Moments moments;
    const bool paired = (p.method == kAntithetic);

    for (std::uint64_t block = first; block < last; ++block) {
        double z[2];
        normals_at(seed, block, z[0], z[1]);

        for (int half = 0; half < 2; ++half) {
            const std::uint64_t unit = 2 * block + half;
            if (unit >= unit_limit) break;
            const double draw = z[half];

            if (paired) {
                // One unit is a pair: the draw and its negative, averaged.
                const double up = p.S * std::exp(p.drift + p.vol * draw);
                const double down = p.S * std::exp(p.drift - p.vol * draw);
                const double value =
                    0.5 * p.discount *
                    (payoff_of(up, p.K, p.kind) + payoff_of(down, p.K, p.kind));
                moments.add(value, 0.0);
            } else if (p.method == kImportance) {
                const double shifted = draw + p.shift;
                const double spot = p.S * std::exp(p.drift + p.vol * shifted);
                const double weight =
                    std::exp(-p.shift * shifted + 0.5 * p.shift * p.shift);
                moments.add(p.discount * payoff_of(spot, p.K, p.kind) * weight, 0.0);
            } else {
                const double spot = p.S * std::exp(p.drift + p.vol * draw);
                const double value = p.discount * payoff_of(spot, p.K, p.kind);
                // The control is the discounted terminal price, mean exactly S.
                moments.add(value, p.method == kControl ? p.discount * spot : 0.0);
            }
        }
    }
    return moments;
}

}  // namespace

// ---------------------------------------------------------------------------
// The C ABI
// ---------------------------------------------------------------------------
//
// Plain C linkage, plain C types, no exceptions crossing the boundary: that is
// what lets a g++-built DLL be called by an MSVC-built Python through ctypes.

extern "C" {

#if defined(_WIN32)
#define MC_EXPORT __declspec(dllexport)
#else
#define MC_EXPORT __attribute__((visibility("default")))
#endif

MC_EXPORT int mc_run(int method, double S, double K, double r, double sigma,
                     double T, int kind, long long n_paths,
                     unsigned long long seed, int n_threads, int n_chunks,
                     double* out_price, double* out_stderr) {
    if (out_price == nullptr || out_stderr == nullptr) return -1;
    if (!(S > 0.0) || !(K > 0.0) || sigma < 0.0 || T < 0.0) return -2;
    if (n_paths < 2) return -3;
    if (method < kPlain || method > kImportance) return -4;
    if (kind < kCall || kind > kStraddle) return -5;
    if (method == kAntithetic && (n_paths % 2) != 0) return -6;
    if (method == kImportance && (!(sigma > 0.0) || !(T > 0.0))) return -7;

    Problem p;
    p.S = S; p.K = K; p.r = r; p.sigma = sigma; p.T = T;
    p.kind = kind; p.method = method;
    p.drift = (r - 0.5 * sigma * sigma) * T;
    p.vol = sigma * std::sqrt(T);
    p.discount = std::exp(-r * T);
    p.shift = 0.0;
    if (method == kImportance) {
        // Put the strike at the median of the sampled distribution: -d2.
        p.shift = -(std::log(S / K) + p.drift) / p.vol;
    }

    // An antithetic run spends n_paths path evaluations as n_paths/2 pairs, so
    // the independent units it averages are the pairs.
    const std::uint64_t units =
        (method == kAntithetic) ? static_cast<std::uint64_t>(n_paths / 2)
                                : static_cast<std::uint64_t>(n_paths);
    if (units < 2) return -3;
    const std::uint64_t blocks = (units + 1) / 2;

    // The chunk count is deliberately NOT the thread count: fixing it keeps the
    // reduction order the same however many threads run, so the answer is
    // bitwise identical across thread counts.
    std::uint64_t chunks = n_chunks > 0 ? static_cast<std::uint64_t>(n_chunks) : 64;
    chunks = std::max<std::uint64_t>(1, std::min<std::uint64_t>(chunks, blocks));
    const std::uint64_t per_chunk = (blocks + chunks - 1) / chunks;

    std::vector<Moments> results(static_cast<std::size_t>(chunks));
    const int threads_wanted = n_threads > 0 ? n_threads : 1;
    const int workers = static_cast<int>(
        std::min<std::uint64_t>(static_cast<std::uint64_t>(threads_wanted), chunks));

    auto worker = [&](int worker_index) {
        for (std::uint64_t c = static_cast<std::uint64_t>(worker_index); c < chunks;
             c += static_cast<std::uint64_t>(workers)) {
            const std::uint64_t first = c * per_chunk;
            const std::uint64_t last = std::min(first + per_chunk, blocks);
            results[static_cast<std::size_t>(c)] =
                run_chunk(p, seed, first, last, units);
        }
    };

    if (workers <= 1) {
        worker(0);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(workers));
        for (int i = 0; i < workers; ++i) pool.emplace_back(worker, i);
        for (auto& thread : pool) thread.join();
    }

    // Combine in chunk order, always.
    Moments total;
    for (const Moments& chunk : results) total.merge(chunk);

    const double n = total.count;
    if (n < 2.0) return -8;

    if (method == kControl) {
        // c* = Cov(Y, X) / Var(X), then price = meanY - c*(meanX - S).
        const double coefficient = total.m2_x > 0.0 ? total.c2_xy / total.m2_x : 0.0;
        *out_price = total.mean_y - coefficient * (total.mean_x - S);
        // Var(Y - c(X - S)) = Var(Y) - 2c Cov(X,Y) + c^2 Var(X), exactly.
        double variance = (total.m2_y - 2.0 * coefficient * total.c2_xy +
                           coefficient * coefficient * total.m2_x) /
                          (n - 1.0);
        if (variance < 0.0) variance = 0.0;   // only reachable by rounding
        *out_stderr = std::sqrt(variance / n);
    } else {
        *out_price = total.mean_y;
        *out_stderr = std::sqrt(total.m2_y / (n - 1.0) / n);
    }
    return 0;
}

MC_EXPORT int mc_hardware_threads() {
    const unsigned int count = std::thread::hardware_concurrency();
    return count == 0 ? 1 : static_cast<int>(count);
}

MC_EXPORT const char* mc_version() { return "derivatives-engine mc_engine 1.0"; }

}  // extern "C"
