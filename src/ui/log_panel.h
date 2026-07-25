// Log panel fed by the core logging sink from any thread.
#pragma once

#include <QWidget>

#include "core/log.h"

class QPlainTextEdit;

namespace sol {

class LogPanel : public QWidget {
    Q_OBJECT

public:
    explicit LogPanel(QWidget* parent = nullptr);
    ~LogPanel() override;

    // Routes core log messages into this panel.
    void installAsLogSink();

signals:
    void messageReceived(int level, const QString& text);

private slots:
    void appendMessage(int level, const QString& text);

private:
    QPlainTextEdit* output_ = nullptr;
};

}  // namespace sol
