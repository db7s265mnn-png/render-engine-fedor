#include "ui/log_panel.h"

#include <QDateTime>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include "ui/theme.h"

namespace sol {

LogPanel::LogPanel(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    output_ = new QPlainTextEdit(this);
    output_->setReadOnly(true);
    output_->setMaximumBlockCount(4000);
    output_->setFont(QFont("monospace", 8));
    layout->addWidget(output_, 1);

    auto* buttons = new QHBoxLayout();
    buttons->addStretch(1);
    auto* clear = new QPushButton("Clear", this);
    connect(clear, &QPushButton::clicked, output_, &QPlainTextEdit::clear);
    buttons->addWidget(clear);
    layout->addLayout(buttons, 0);

    connect(this, &LogPanel::messageReceived, this, &LogPanel::appendMessage, Qt::QueuedConnection);
}

LogPanel::~LogPanel() { setLogSink(nullptr); }

void LogPanel::installAsLogSink() {
    setLogSink([this](LogLevel level, const std::string& message) {
        // Emitted from worker threads; the queued connection marshals it.
        emit messageReceived(int(level), QString::fromStdString(message));
    });
}

void LogPanel::appendMessage(int level, const QString& text) {
    QString color = "#dcdee2";
    switch (LogLevel(level)) {
        case LogLevel::Debug: color = "#8a8f98"; break;
        case LogLevel::Info: color = "#dcdee2"; break;
        case LogLevel::Warning: color = "#ffc85a"; break;
        case LogLevel::Error: color = "#dc5a50"; break;
    }
    const QString time = QDateTime::currentDateTime().toString("hh:mm:ss");
    output_->appendHtml(QString("<span style='color:#6f747c'>%1</span> <span style='color:%2'>%3</span>")
                            .arg(time, color, text.toHtmlEscaped()));
}

}  // namespace sol
