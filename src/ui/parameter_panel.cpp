#include "ui/parameter_panel.h"

#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMimeData>
#include <QPushButton>
#include <QScrollArea>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QVector3D>
#include <functional>

#include "nodes/node_registry.h"
#include "ui/theme.h"

namespace sol {
namespace {

constexpr const char* kPrimPathMime = "application/x-fedor-prim-path";

// String editor that accepts prim paths dragged from the scene graph and
// commits immediately on drop / paste so Assign To updates without leaving the field.
class PathLineEdit : public QLineEdit {
public:
    using QLineEdit::QLineEdit;
    std::function<void(const QString&)> onCommitted;

protected:
    void dragEnterEvent(QDragEnterEvent* event) override {
        if (event->mimeData()->hasFormat(kPrimPathMime) || event->mimeData()->hasText()) {
            event->acceptProposedAction();
            return;
        }
        QLineEdit::dragEnterEvent(event);
    }

    void dragMoveEvent(QDragMoveEvent* event) override {
        if (event->mimeData()->hasFormat(kPrimPathMime) || event->mimeData()->hasText()) {
            event->acceptProposedAction();
            return;
        }
        QLineEdit::dragMoveEvent(event);
    }

    void dropEvent(QDropEvent* event) override {
        QString path;
        if (event->mimeData()->hasFormat(kPrimPathMime))
            path = QString::fromUtf8(event->mimeData()->data(kPrimPathMime));
        else if (event->mimeData()->hasText())
            path = event->mimeData()->text().trimmed();
        // Keep only the first line — clipboard/drag payloads can include extras.
        const int newline = path.indexOf('\n');
        if (newline >= 0) path = path.left(newline).trimmed();
        if (!path.isEmpty()) {
            setText(path);
            if (onCommitted) onCommitted(path);
            event->acceptProposedAction();
            return;
        }
        QLineEdit::dropEvent(event);
    }

    void keyPressEvent(QKeyEvent* event) override {
        QLineEdit::keyPressEvent(event);
        if (event->matches(QKeySequence::Paste) && onCommitted) onCommitted(text());
    }
};

QColor linearToDisplayColor(Vec3 c) {
    return QColor::fromRgbF(double(linearToSrgb(c.x)), double(linearToSrgb(c.y)), double(linearToSrgb(c.z)));
}

Vec3 displayColorToLinear(const QColor& color) {
    return Vec3(srgbToLinear(float(color.redF())), srgbToLinear(float(color.greenF())),
                srgbToLinear(float(color.blueF())));
}

QDoubleSpinBox* makeDoubleSpin(double value, double minimum, double maximum) {
    auto* spin = new QDoubleSpinBox();
    spin->setRange(minimum, maximum);
    spin->setDecimals(4);
    spin->setSingleStep(std::max(0.001, (maximum - minimum) / 200.0));
    spin->setValue(value);
    spin->setKeyboardTracking(false);
    spin->setMinimumWidth(70);
    return spin;
}

}  // namespace

ParameterPanel::ParameterPanel(QWidget* parent) : QWidget(parent) {
    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);

    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    outer->addWidget(scroll);

    content_ = new QWidget();
    contentLayout_ = new QVBoxLayout(content_);
    contentLayout_->setContentsMargins(8, 8, 8, 8);
    contentLayout_->setSpacing(8);
    scroll->setWidget(content_);

    rebuild();
}

void ParameterPanel::setNode(Node* node) {
    if (node_ == node) return;
    node_ = node;
    rebuild();
}

void ParameterPanel::refresh() { rebuild(); }

void ParameterPanel::rebuild() {
    updating_ = true;
    QLayoutItem* child = nullptr;
    while ((child = contentLayout_->takeAt(0)) != nullptr) {
        if (child->widget()) child->widget()->deleteLater();
        delete child;
    }
    nameEdit_ = nullptr;

    if (!node_) {
        auto* hint = new QLabel("No node selected.\n\nPress Tab in the network editor to add one.");
        hint->setStyleSheet("color: #969aa0;");
        hint->setWordWrap(true);
        contentLayout_->addWidget(hint);
        contentLayout_->addStretch(1);
        updating_ = false;
        return;
    }

    // Header: node name plus type description.
    auto* header = new QGroupBox(node_->typeName());
    auto* headerLayout = new QFormLayout(header);
    nameEdit_ = new QLineEdit(node_->name());
    connect(nameEdit_, &QLineEdit::editingFinished, this, [this] {
        if (!node_ || updating_) return;
        const QString text = nameEdit_->text().trimmed();
        if (!text.isEmpty() && text != node_->name()) {
            node_->setName(text);
            emit nodeRenamed(node_);
        }
    });
    headerLayout->addRow("Name", nameEdit_);

    if (const NodeTypeInfo* info = NodeRegistry::instance().find(node_->typeName())) {
        auto* description = new QLabel(info->description);
        description->setWordWrap(true);
        description->setStyleSheet("color: #969aa0;");
        headerLayout->addRow(description);
    }
    if (!node_->errorText().isEmpty()) {
        auto* error = new QLabel(node_->errorText());
        error->setWordWrap(true);
        error->setStyleSheet("color: #dc5a50;");
        headerLayout->addRow("Error", error);
    }
    contentLayout_->addWidget(header);

    // One group box per parameter folder, ungrouped parameters first.
    QStringList groups;
    groups << QString();
    for (const Parameter& parameter : node_->parameters()) {
        if (!groups.contains(parameter.group)) groups << parameter.group;
    }

    for (const QString& group : groups) {
        QFormLayout* form = nullptr;
        for (Parameter& parameter : node_->parameters()) {
            if (parameter.group != group) continue;
            if (!form) {
                auto* box = new QGroupBox(group.isEmpty() ? QString("Parameters") : group);
                form = new QFormLayout(box);
                form->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);
                contentLayout_->addWidget(box);
            }
            QWidget* editor = createEditor(parameter);
            if (!editor) continue;
            if (!parameter.tooltip.isEmpty()) editor->setToolTip(parameter.tooltip);
            form->addRow(parameter.label, editor);
        }
    }

    contentLayout_->addStretch(1);
    updating_ = false;
}

QWidget* ParameterPanel::createEditor(Parameter& parameter) {
    const QString name = parameter.name;
    Node* node = node_;
    auto notify = [this, node, name](const QVariant& value) {
        if (updating_ || !node) return;
        node->setParameterValue(name, value);
        emit parameterEdited(node, name);
    };

    switch (parameter.type) {
        case ParamType::Float: {
            auto* container = new QWidget();
            auto* layout = new QHBoxLayout(container);
            layout->setContentsMargins(0, 0, 0, 0);
            auto* spin = makeDoubleSpin(parameter.toDouble(), parameter.hasRange ? parameter.minValue : -1e7,
                                        parameter.hasRange ? parameter.maxValue : 1e7);
            QSlider* slider = nullptr;
            if (parameter.hasRange) {
                slider = new QSlider(Qt::Horizontal);
                slider->setRange(0, 1000);
                const double t = (parameter.toDouble() - parameter.minValue) /
                                 std::max(1e-9, parameter.maxValue - parameter.minValue);
                slider->setValue(int(t * 1000.0));
                layout->addWidget(slider, 1);
            }
            layout->addWidget(spin, 0);

            connect(spin, &QDoubleSpinBox::valueChanged, this, [notify, slider, parameter](double value) {
                if (slider) {
                    QSignalBlocker blocker(slider);
                    const double t = (value - parameter.minValue) /
                                     std::max(1e-9, parameter.maxValue - parameter.minValue);
                    slider->setValue(int(t * 1000.0));
                }
                notify(value);
            });
            if (slider) {
                connect(slider, &QSlider::valueChanged, this, [spin, parameter](int value) {
                    const double v = parameter.minValue +
                                     (parameter.maxValue - parameter.minValue) * (double(value) / 1000.0);
                    spin->setValue(v);
                });
            }
            return container;
        }
        case ParamType::Int: {
            auto* spin = new QSpinBox();
            spin->setRange(parameter.hasRange ? int(parameter.minValue) : -1000000,
                           parameter.hasRange ? int(parameter.maxValue) : 1000000);
            spin->setValue(parameter.toInt());
            spin->setKeyboardTracking(false);
            connect(spin, &QSpinBox::valueChanged, this, [notify](int value) { notify(value); });
            return spin;
        }
        case ParamType::Bool: {
            auto* check = new QCheckBox();
            check->setChecked(parameter.toBool());
            connect(check, &QCheckBox::toggled, this, [notify](bool value) { notify(value); });
            return check;
        }
        case ParamType::Vec3: {
            auto* container = new QWidget();
            auto* layout = new QHBoxLayout(container);
            layout->setContentsMargins(0, 0, 0, 0);
            const Vec3 value = parameter.toVec3();
            QDoubleSpinBox* spins[3];
            for (int i = 0; i < 3; ++i) {
                spins[i] = makeDoubleSpin(double(value[i]), -1e6, 1e6);
                spins[i]->setDecimals(3);
                layout->addWidget(spins[i]);
            }
            for (int i = 0; i < 3; ++i) {
                connect(spins[i], &QDoubleSpinBox::valueChanged, this,
                        [notify, spins](double) {
                            notify(QVariant::fromValue(QVector3D(float(spins[0]->value()),
                                                                 float(spins[1]->value()),
                                                                 float(spins[2]->value()))));
                        });
            }
            return container;
        }
        case ParamType::Color: {
            auto* container = new QWidget();
            auto* layout = new QHBoxLayout(container);
            layout->setContentsMargins(0, 0, 0, 0);
            auto* button = new QPushButton();
            button->setMinimumWidth(48);
            const Vec3 value = parameter.toVec3();
            const QColor initial = linearToDisplayColor(value);
            button->setStyleSheet(QString("background-color: %1; border: 1px solid #22242a;").arg(initial.name()));
            layout->addWidget(button, 1);

            QDoubleSpinBox* spins[3];
            for (int i = 0; i < 3; ++i) {
                spins[i] = makeDoubleSpin(double(value[i]), 0.0, 100.0);
                spins[i]->setDecimals(3);
                layout->addWidget(spins[i]);
            }
            auto pushValue = [notify, spins] {
                notify(QVariant::fromValue(
                    QVector3D(float(spins[0]->value()), float(spins[1]->value()), float(spins[2]->value()))));
            };
            for (int i = 0; i < 3; ++i) {
                connect(spins[i], &QDoubleSpinBox::valueChanged, this, [pushValue, button, spins](double) {
                    const Vec3 linear(float(spins[0]->value()), float(spins[1]->value()), float(spins[2]->value()));
                    button->setStyleSheet(QString("background-color: %1; border: 1px solid #22242a;")
                                              .arg(linearToDisplayColor(linear).name()));
                    pushValue();
                });
            }
            connect(button, &QPushButton::clicked, this, [this, spins, button] {
                const Vec3 current(float(spins[0]->value()), float(spins[1]->value()), float(spins[2]->value()));
                const QColor chosen = QColorDialog::getColor(linearToDisplayColor(current), this, "Pick colour");
                if (!chosen.isValid()) return;
                const Vec3 linear = displayColorToLinear(chosen);
                for (int i = 0; i < 3; ++i) spins[i]->setValue(double(linear[i]));
                button->setStyleSheet(
                    QString("background-color: %1; border: 1px solid #22242a;").arg(chosen.name()));
            });
            return container;
        }
        case ParamType::String: {
            auto* edit = new PathLineEdit(parameter.toString());
            edit->setAcceptDrops(true);
            if (parameter.name == "pattern") {
                edit->setPlaceholderText("e.g. /geo/sphere1 — Ctrl+V or drop from Scene Graph");
                if (parameter.tooltip.isEmpty())
                    edit->setToolTip("Prim path or glob. Copy a prim with Ctrl+C in the Scene Graph, "
                                     "or drag its name into this field.");
            }
            edit->onCommitted = [notify](const QString& text) { notify(text); };
            connect(edit, &QLineEdit::editingFinished, this, [notify, edit] { notify(edit->text()); });
            return edit;
        }
        case ParamType::FilePath: {
            auto* container = new QWidget();
            auto* layout = new QHBoxLayout(container);
            layout->setContentsMargins(0, 0, 0, 0);
            auto* edit = new QLineEdit(parameter.toString());
            auto* browse = new QPushButton("...");
            browse->setFixedWidth(30);
            layout->addWidget(edit, 1);
            layout->addWidget(browse, 0);
            const QString filter = parameter.fileFilter;
            connect(edit, &QLineEdit::editingFinished, this, [notify, edit] { notify(edit->text()); });
            connect(browse, &QPushButton::clicked, this, [this, edit, filter, notify] {
                const QString path = QFileDialog::getOpenFileName(this, "Choose file", edit->text(), filter);
                if (path.isEmpty()) return;
                edit->setText(path);
                notify(path);
            });
            return container;
        }
        case ParamType::Menu: {
            auto* combo = new QComboBox();
            combo->addItems(parameter.menuItems);
            combo->setCurrentIndex(parameter.toInt());
            connect(combo, &QComboBox::currentIndexChanged, this, [notify](int index) { notify(index); });
            return combo;
        }
        case ParamType::Label: {
            auto* label = new QLabel(parameter.toString());
            label->setWordWrap(true);
            return label;
        }
    }
    return nullptr;
}

}  // namespace sol
