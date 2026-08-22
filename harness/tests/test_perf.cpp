#include "test_common.hpp"

#include "micro_llm/micro_llm.hpp"

#include <chrono>
#include <string>
#include <thread>
#include <vector>

void test_perf_clocks(TestContext& ctx) {
    using namespace micro_llm;

    PerfClocks clocks;
    clocks.begin_session();
    clocks.set_plan(57, 7, true);
    clocks.begin_span(PerfSpan::Gpu);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    clocks.end_span(PerfSpan::Gpu);
    clocks.begin_span(PerfSpan::Pcie);
    clocks.add_h2d(150u * 1024u * 1024u);
    clocks.end_span(PerfSpan::Pcie);
    clocks.add_d2h(kFloorBitsetBytes);
    clocks.add_cuda_ffn_bind();
    clocks.add_overlap_prefetch();
    clocks.end_token();

    const PerfSnapshot s = clocks.snapshot();
    CHECK(ctx, s.n_tokens == 1);
    CHECK(ctx, s.n_parked_ffn == 57);
    CHECK(ctx, s.n_streamed_ffn == 7);
    CHECK(ctx, s.h2d_bytes_per_tok == 150ull * 1024ull * 1024ull);
    CHECK(ctx, s.d2h_bytes_per_tok == kFloorBitsetBytes);
    CHECK(ctx, s.d2h_bytes_per_tok < 200000);  // ~140KB, not 17408*4*64
    CHECK(ctx, s.cuda_ffn_binds == 1);
    CHECK(ctx, s.host_ffn_binds == 0);
    CHECK(ctx, s.gpu_ms > 0.0);
    CHECK(ctx, s.host_pages_pinned);

    const std::string line = format_performance_line(s);
    CHECK(ctx, line.find("PERFORMANCE") == 0);
    CHECK(ctx, line.find("tok/s=") != std::string::npos);
    CHECK(ctx, line.find("gpu_ms=") != std::string::npos);
    CHECK(ctx, line.find("pcie_ms=") != std::string::npos);
    CHECK(ctx, line.find("h2d_B=") != std::string::npos);
    CHECK(ctx, line.find("vram_") != std::string::npos);

    const std::string bots = format_performance_bottlenecks(s);
    CHECK(ctx, bots.find("PERFORMANCE bottlenecks=") == 0);

    PerfSnapshot host;
    host.host_ffn_binds = 64;
    host.cuda_ffn_binds = 0;
    host.n_streamed_ffn = 64;
    host.cpu_ms = 4.5;
    CHECK(ctx, std::string(top_bottleneck(host)) == "host_ffn");

    HookRing ring;
    std::vector<uint8_t> rec(kHtr1RecordBytes, 0);
    rec[0] = 7;
    CHECK(ctx, ring.push(rec.data()) < kHtr1RingDepth);
    std::vector<uint8_t> out(kHtr1RecordBytes, 0);
    CHECK(ctx, ring.copy_latest(out.data()));
    CHECK(ctx, out[0] == 7);
}
