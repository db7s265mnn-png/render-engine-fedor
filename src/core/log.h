// Minimal thread safe logging facility. The UI installs a sink to mirror
// messages into the log panel; the CLI prints to stderr.
#pragma once

#include <functional>
#include <string>

namespace sol {

enum class LogLevel { Debug, Info, Warning, Error };

using LogSink = std::function<void(LogLevel, const std::string&)>;

void setLogSink(LogSink sink);
void setLogLevel(LogLevel minimumLevel);
void logMessage(LogLevel level, const std::string& message);

inline void logDebug(const std::string& m) { logMessage(LogLevel::Debug, m); }
inline void logInfo(const std::string& m) { logMessage(LogLevel::Info, m); }
inline void logWarning(const std::string& m) { logMessage(LogLevel::Warning, m); }
inline void logError(const std::string& m) { logMessage(LogLevel::Error, m); }

const char* logLevelName(LogLevel level);

}  // namespace sol
