// Numeric editors that ignore the mouse wheel unless the control was clicked
// (has keyboard focus), and always use '.' as the decimal separator.
#pragma once

#include <QAbstractSpinBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLocale>
#include <QSlider>
#include <QSpinBox>
#include <QWheelEvent>

namespace sol {

inline void prepareNumericEditor(QWidget* widget) {
    if (!widget) return;
    widget->setFocusPolicy(Qt::StrongFocus);
    if (auto* spin = qobject_cast<QAbstractSpinBox*>(widget)) spin->setLocale(QLocale::c());
}

class NoWheelDoubleSpinBox : public QDoubleSpinBox {
public:
    explicit NoWheelDoubleSpinBox(QWidget* parent = nullptr) : QDoubleSpinBox(parent) {
        prepareNumericEditor(this);
    }

protected:
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
    }

protected:
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
