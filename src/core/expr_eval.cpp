#include "core/expr_eval.h"

#include <cmath>
#include <string>

namespace sol {
namespace {

thread_local int g_exprFrame = 1;

struct ParseState {
    std::string s;
    size_t i = 0;
    QString error;
};

void skipWs(ParseState& st) {
    while (st.i < st.s.size() && (st.s[st.i] == ' ' || st.s[st.i] == '\t')) ++st.i;
}

bool parseExpr(ParseState& st, double& out);

bool parsePrimary(ParseState& st, double& out) {
    skipWs(st);
    if (st.i >= st.s.size()) {
        st.error = QStringLiteral("unexpected end of expression");
        return false;
    }
    if (st.s[st.i] == '+') {
        ++st.i;
        return parsePrimary(st, out);
    }
    if (st.s[st.i] == '-') {
        ++st.i;
        double v = 0.0;
        if (!parsePrimary(st, v)) return false;
        out = -v;
        return true;
    }
    if (st.s[st.i] == '(') {
        ++st.i;
        if (!parseExpr(st, out)) return false;
        skipWs(st);
        if (st.i >= st.s.size() || st.s[st.i] != ')') {
            st.error = QStringLiteral("missing ')'");
            return false;
        }
        ++st.i;
        return true;
    }
    // Number
    size_t start = st.i;
    if (st.s[st.i] == '.' || (st.s[st.i] >= '0' && st.s[st.i] <= '9')) {
        bool sawDot = false;
        while (st.i < st.s.size()) {
            const char c = st.s[st.i];
            if (c >= '0' && c <= '9') {
                ++st.i;
                continue;
            }
            if (c == '.' && !sawDot) {
                sawDot = true;
                ++st.i;
                continue;
            }
            if ((c == 'e' || c == 'E') && st.i + 1 < st.s.size()) {
                ++st.i;
                if (st.s[st.i] == '+' || st.s[st.i] == '-') ++st.i;
                while (st.i < st.s.size() && st.s[st.i] >= '0' && st.s[st.i] <= '9') ++st.i;
                break;
            }
            break;
        }
        try {
            out = std::stod(st.s.substr(start, st.i - start));
            return true;
        } catch (...) {
            st.error = QStringLiteral("invalid number");
            return false;
        }
    }
    st.error = QStringLiteral("unexpected character '%1'").arg(QChar(st.s[st.i]));
    return false;
}

bool parsePower(ParseState& st, double& out) {
    if (!parsePrimary(st, out)) return false;
    skipWs(st);
    if (st.i < st.s.size() && st.s[st.i] == '^') {
        ++st.i;
        double rhs = 0.0;
        if (!parsePower(st, rhs)) return false;  // right-assoc
        out = std::pow(out, rhs);
    }
    return true;
}

bool parseTerm(ParseState& st, double& out) {
    if (!parsePower(st, out)) return false;
    for (;;) {
        skipWs(st);
        if (st.i >= st.s.size()) break;
        const char op = st.s[st.i];
        if (op != '*' && op != '/') break;
        ++st.i;
        double rhs = 0.0;
        if (!parsePower(st, rhs)) return false;
        if (op == '*') out *= rhs;
        else {
            if (rhs == 0.0) {
                st.error = QStringLiteral("division by zero");
                return false;
            }
            out /= rhs;
        }
    }
    return true;
}

bool parseExpr(ParseState& st, double& out) {
    if (!parseTerm(st, out)) return false;
    for (;;) {
        skipWs(st);
        if (st.i >= st.s.size()) break;
        const char op = st.s[st.i];
        if (op != '+' && op != '-') break;
        ++st.i;
        double rhs = 0.0;
        if (!parseTerm(st, rhs)) return false;
        if (op == '+') out += rhs;
        else out -= rhs;
    }
    return true;
}

}  // namespace

void setExprFrame(int frame) { g_exprFrame = frame; }
int exprFrame() { return g_exprFrame; }

QString expandFrameTokens(const QString& text, int frame) {
    QString out = text;
    // Longest first so $F4 wins over $F.
    for (int width = 8; width >= 2; --width) {
        const QString token = QStringLiteral("$F%1").arg(width);
        const QString repl = QStringLiteral("%1").arg(frame, width, 10, QChar('0'));
        out.replace(token, repl);
    }
    out.replace(QStringLiteral("$F"), QString::number(frame));
    return out;
}

bool looksLikeExpression(const QString& text) {
    const QString t = text.trimmed();
    if (t.isEmpty()) return false;
    if (t.contains(QLatin1Char('$'))) return true;
    // Plain number (optional sign / decimal / scientific) is not an expression.
    bool ok = false;
    t.toDouble(&ok);
    if (ok) {
        // toDouble accepts "1e3" etc.; reject if any unexpected letters remain besides e/E.
        for (QChar c : t) {
            if (c.isLetter() && c != QLatin1Char('e') && c != QLatin1Char('E')) return true;
        }
        return false;
    }
    // Non-numeric with operators → expression attempt.
    return t.contains(QLatin1Char('+')) || t.contains(QLatin1Char('*')) || t.contains(QLatin1Char('/')) ||
           t.contains(QLatin1Char('^')) || t.contains(QLatin1Char('('));
}

bool evalExpression(const QString& text, int frame, double& out, QString* error) {
    const QString expanded = expandFrameTokens(text.trimmed(), frame);
    // Fast path: plain number after expansion.
    bool ok = false;
    const double asNum = expanded.toDouble(&ok);
    if (ok) {
        // Ensure entire string was a number (no trailing junk).
        QString check = expanded.trimmed();
        // Allow leading +
        out = asNum;
        // Verify by re-parsing with our parser for consistency on "1+2".
        if (!check.contains(QLatin1Char('+')) && !check.contains(QLatin1Char('*')) &&
            !check.contains(QLatin1Char('/')) && !check.contains(QLatin1Char('^')) &&
            !check.contains(QLatin1Char('(')) &&
            !(check.count(QLatin1Char('-')) > 1 ||
              (check.count(QLatin1Char('-')) == 1 && !check.startsWith(QLatin1Char('-'))))) {
            return true;
        }
    }

    ParseState st;
    st.s = expanded.toStdString();
    if (!parseExpr(st, out)) {
        if (error) *error = st.error.isEmpty() ? QStringLiteral("invalid expression") : st.error;
        return false;
    }
    skipWs(st);
    if (st.i != st.s.size()) {
        if (error) *error = QStringLiteral("trailing characters in expression");
        return false;
    }
    if (!std::isfinite(out)) {
        if (error) *error = QStringLiteral("non-finite result");
        return false;
    }
    return true;
}

QString expandStringExpression(const QString& text, int frame) {
    return expandFrameTokens(text, frame);
}

QString expressionFieldStyleSheet() {
    return QStringLiteral(
        "QLineEdit, QSpinBox, QDoubleSpinBox {"
        "  background-color: #1e3a28;"
        "  color: #d8f5d8;"
        "  border: 1px solid #5cc45c;"
        "  border-radius: 2px;"
        "  selection-background-color: #3a7a4a;"
        "}");
}

QString normalFieldStyleSheet() { return QString(); }

}  // namespace sol
