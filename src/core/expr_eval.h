// Lightweight Houdini-style expressions: $F token expand + math (+ - * / ^ ()).
#pragma once

#include <QString>

namespace sol {

// Current global frame used when evaluating expressions (set from timeline / TX scrubber).
void setExprFrame(int frame);
int exprFrame();

// True when text should be treated as an expression (not a plain numeric / plain path literal).
bool looksLikeExpression(const QString& text);

// Expand $F to the current frame number. Does not touch <UDIM>.
QString expandFrameTokens(const QString& text, int frame);

// Evaluate a numeric expression after frame-token expansion.
// Supports + - * / ^ (power) and parentheses. Returns false on error.
bool evalExpression(const QString& text, int frame, double& out, QString* error = nullptr);

// Expand frame tokens in a path/string (for FilePath / String parameters).
QString expandStringExpression(const QString& text, int frame);

// Resolve a path containing $F: tries unpadded and common zero-paddings to find a file on disk.
QString resolveFramePathExisting(const QString& text, int frame);

// Houdini-like green chrome for expression fields.
QString expressionFieldStyleSheet();
QString normalFieldStyleSheet();

}  // namespace sol
