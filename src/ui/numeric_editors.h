// Numeric editors that ignore the mouse wheel unless the control was clicked
// (has keyboard focus), and always use '.' as the decimal separator.
// FreeFloatLineEdit allows placing the caret anywhere and typing freely.
#pragma once

#include <QAbstractSpinBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFocusEvent>
#include <QLineEdit>
#include <QLocale>
#include <QMouseEvent>
#include <QSlider>
#include <QSpinBox>
#include <QWheelEvent>
#include <cmath>

namespace sol {

inline void prepareNumericEditor(QWidget* widget) {
    if (!widget) return;
    widget->setFocusPolicy(Qt::StrongFocus);
    if (auto* spin = qobject_cast<QAbstractSpinBox*>(widget)) spin->setLocale(QLocale::c());
}

// Plain line edit for float authoring — caret can sit between any digits.
class FreeFloatLineEdit : public QLineEdit {
public:
    explicit FreeFloatLineEdit(double value, QWidget* parent = nullptr) : QLineEdit(parent) {
        setLocale(QLocale::c());
        setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        setFocusPolicy(Qt::StrongFocus);
        setValue(value);
    }

    void setValue(double value, bool force = false) {
        // Never clobber in-progress typing unless the caller just committed.
        if (hasFocus() && !force) return;
        const QString text = QString::number(value, 'g', 9);
        if (this->text() == text) return;
        setText(text);
    }

    double value(bool* ok = nullptr) const {
        bool parsed = false;
        const double v = QLocale::c().toDouble(text().trimmed(), &parsed);
        if (ok) *ok = parsed && std::isfinite(v);
        return parsed ? v : 0.0;
    }

protected:
    void wheelEvent(QWheelEvent* event) override {
        if (!hasFocus()) {
            event->ignore();
            return;
        }
        QLineEdit::wheelEvent(event);
    }

    void mousePressEvent(QMouseEvent* event) override {
        // Ensure the first click places a caret instead of only focusing the widget.
        if (!hasFocus()) setFocus(Qt::MouseFocusReason);
        QLineEdit::mousePressEvent(event);
    }
};

class NoWheelDoubleSpinBox : public QDoubleSpinBox {
public:
    explicit NoWheelDoubleSpinBox(QWidget* parent = nullptr) : QDoubleSpinBox(parent) {
        prepareNumericEditor(this);
        setKeyboardTracking(false);
        setCorrectionMode(QAbstractSpinBox::CorrectToNearestValue);
        if (QLineEdit* edit = lineEdit()) {
            edit->setFocusPolicy(Qt::StrongFocus);
            // Avoid Select-All on every click so the caret can sit between digits.
            edit->setCursorPosition(edit->text().size());
        }
    }

protected:
    void focusInEvent(QFocusEvent* event) override {
        QDoubleSpinBox::focusInEvent(event);
        if (event->reason() == Qt::MouseFocusReason) {
            if (QLineEdit* edit = lineEdit()) {
                // Qt selects-all on some styles; clear that so a click can place the caret.
                const int pos = edit->cursorPosition();
                edit->deselect();
                edit->setCursorPosition(pos);
            }
        }
    }

    void mousePressEvent(QMouseEvent* event) override {
        if (!hasFocus()) setFocus(Qt::MouseFocusReason);
        QDoubleSpinBox::mousePressEvent(event);
        if (QLineEdit* edit = lineEdit()) {
            // Map click into the embedded line edit for precise caret placement.
            const QPoint local = edit->mapFrom(this, event->pos());
            if (edit->rect().contains(local)) {
                const int pos = edit->cursorPositionAt(local);
                edit->deselect();
                edit->setCursorPosition(pos);
            }
        }
    }

    void wheelEvent(QWheelEvent* event) override {
        if (!hasFocus()) {
            event->ignore();
            return;
        }
        QDoubleSpinBox::wheelEvent(event);
    }
};

class NoWheelSpinBox : public QSpinBox {
public:
    explicit NoWheelSpinBox(QWidget* parent = nullptr) : QSpinBox(parent) {
        prepareNumericEditor(this);
        setKeyboardTracking(false);
        if (QLineEdit* edit = lineEdit()) edit->setFocusPolicy(Qt::StrongFocus);
    }

protected:
    void focusInEvent(QFocusEvent* event) override {
        QSpinBox::focusInEvent(event);
        if (event->reason() == Qt::MouseFocusReason) {
            if (QLineEdit* edit = lineEdit()) {
                const int pos = edit->cursorPosition();
                edit->deselect();
                edit->setCursorPosition(pos);
            }
        }
    }

    void mousePressEvent(QMouseEvent* event) override {
        if (!hasFocus()) setFocus(Qt::MouseFocusReason);
        QSpinBox::mousePressEvent(event);
        if (QLineEdit* edit = lineEdit()) {
            const QPoint local = edit->mapFrom(this, event->pos());
            if (edit->rect().contains(local)) {
                const int pos = edit->cursorPositionAt(local);
                edit->deselect();
                edit->setCursorPosition(pos);
            }
        }
    }

    void wheelEvent(QWheelEvent* event) override {
        if (!hasFocus()) {
            event->ignore();
            return;
        }
        QSpinBox::wheelEvent(event);
    }
};

class NoWheelSlider : public QSlider {
public:
    using QSlider::QSlider;

    explicit NoWheelSlider(Qt::Orientation orientation, QWidget* parent = nullptr)
        : QSlider(orientation, parent) {
        setFocusPolicy(Qt::StrongFocus);
    }

protected:
    void wheelEvent(QWheelEvent* event) override {
        if (!hasFocus()) {
            event->ignore();
            return;
        }
        QSlider::wheelEvent(event);
    }
};

class NoWheelComboBox : public QComboBox {
public:
    explicit NoWheelComboBox(QWidget* parent = nullptr) : QComboBox(parent) {
        setFocusPolicy(Qt::StrongFocus);
    }

protected:
    void wheelEvent(QWheelEvent* event) override {
        if (!hasFocus()) {
            event->ignore();
            return;
        }
        QComboBox::wheelEvent(event);
    }
};

}  // namespace sol
