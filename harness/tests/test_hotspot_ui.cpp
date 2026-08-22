#include "micro_llm/hotspot_ui.hpp"
#include "test_common.hpp"

#include <fstream>
#include <string>

void test_hotspot_ui_files(TestContext& ctx) {
    std::string err;
    const std::string dir = micro_llm::hotspot_ui_dir(&err);
    CHECK(ctx, !dir.empty());
    CHECK(ctx, err.empty());
    if (dir.empty()) return;

    std::ifstream index(dir + "/index.html");
    CHECK(ctx, index.good());
    std::string html;
    std::string line;
    while (std::getline(index, line)) {
        html += line;
        html += '\n';
    }
    CHECK(ctx, html.find("stage") != std::string::npos);
    CHECK(ctx, html.find("15.2 labeled example") != std::string::npos);
    CHECK(ctx, html.find("id=\"out\"") == std::string::npos);

    micro_llm::HotspotUiOptions opt;
    opt.check_only = true;
    CHECK(ctx, micro_llm::run_hotspot_ui(opt) == 0);
}
