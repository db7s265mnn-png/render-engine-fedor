#include "core/log.h"

#include <cstdio>
#include <mutex>

namespace sol {
namespace {
std::mutex g_mutex;
LogSink g_sink;
LogLevel g_minLevel = LogLevel::Info;
}  // namespace

void setLogSink(LogSink sink) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_sink = std::move(sink);
}

void setLogLevel(LogLevel minimumLevel) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_minLevel = minimumLevel;
}

const char* logLevelName(LogLevel level) {
    switch (level) {
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info: return "INFO";
        case LogLevel::Warning: return "WARNING";
        case LogLevel::Error: return "ERROR";
    }
    return "INFO";
}

void logMessage(LogLevel level, const std::string& message) {
    LogSink sink;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (static_cast<int>(level) < static_cast<int>(g_minLevel)) return;
        sink = g_sink;
    }
    if (sink) {
        sink(level, message);
    } else {
        std::fprintf(stderr, "[%s] %s\n", logLevelName(level), message.c_str());
        std::fflush(stderr);
    }
}

}  // namespace sol
