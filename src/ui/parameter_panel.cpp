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
#include <QRegularExpression>
#include <QScrollArea>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QVector3D>
#include <functional>

#include "nodes/node_registry.h"
#include "ui/numeric_editors.h"
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

NoWheelDoubleSpinBox* makeDoubleSpin(double value, double minimum, double maximum) {
    auto* spin = new NoWheelDoubleSpinBox();
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

void ParameterPanel::clearSelection() {
    node_ = nullptr;
    materialXMode_ = false;
    materialX_ = {};
    rebuild();
}

void ParameterPanel::setNode(Node* node) {
    if (!materialXMode_ && node_ == node) return;
    node_ = node;
    materialXMode_ = false;
    materialX_ = {};
    rebuild();
}

void ParameterPanel::setMaterialXSelection(const MaterialXSelection& selection) {
    node_ = selection.hostMaterial;
    materialXMode_ = selection.hostMaterial && !selection.name.isEmpty();
    materialX_ = selection;
    if (!materialXMode_) {
        materialX_ = {};
        rebuild();
        return;
    }
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

    if (materialXMode_) {
        rebuildMaterialX();
        updating_ = false;
        return;
    }

    if (!node_) {
        auto* hint = new QLabel("No node selected.\n\nPress Tab in the network editor to add one,\n"
                                "or select a node in the Material Network.");
        hint->setStyleSheet("color: #969aa0;");
        hint->setWordWrap(true);
        contentLayout_->addWidget(hint);
        contentLayout_->addStretch(1);
        updating_ = false;
        return;
    }

    rebuildLop();
    updating_ = false;
}

void ParameterPanel::rebuildLop() {
    // Header: node name plus type description.
    auto* header = new QGroupBox(node_->typeName());
    auto* headerLayout = new QFormLayout(header);
    nameEdit_ = new QLineEdit(node_->name());
    connect(nameEdit_, &QLineEdit::editingFinished, this, [this] {
        if (!node_ || updating_ || materialXMode_) return;
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
        if (parameter.name == "mtlx") continue;
        if (!groups.contains(parameter.group)) groups << parameter.group;
    }

    for (const QString& group : groups) {
        QFormLayout* form = nullptr;
        for (Parameter& parameter : node_->parameters()) {
            if (parameter.name == "mtlx") continue;
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
}

void ParameterPanel::rebuildMaterialX() {
    auto* header = new QGroupBox("MaterialX Node");
    auto* headerLayout = new QFormLayout(header);
    auto* typeLabel = new QLabel(materialX_.category +
                                 (materialX_.type.isEmpty() ? QString() : ("  ·  " + materialX_.type)));
    typeLabel->setStyleSheet("color: #969aa0;");
    typeLabel->setWordWrap(true);
    headerLayout->addRow("Type", typeLabel);

    nameEdit_ = new QLineEdit(materialX_.name);
    nameEdit_->setPlaceholderText("node name");
    connect(nameEdit_, &QLineEdit::editingFinished, this, [this] {
        if (updating_ || !materialXMode_ || !materialX_.hostMaterial) return;
        const QString text = nameEdit_->text().trimmed();
        if (text.isEmpty() || text == materialX_.name) return;
        emit materialXRenamed(materialX_.hostMaterial, materialX_.name, text);
    });
    headerLayout->addRow("Name", nameEdit_);
    if (materialX_.hostMaterial) {
        auto* host = new QLabel(materialX_.hostMaterial->name());
        host->setStyleSheet("color: #969aa0;");
        headerLayout->addRow("Material", host);
    }
    contentLayout_->addWidget(header);

    auto* paramsBox = new QGroupBox("Parameters");
    auto* form = new QFormLayout(paramsBox);
    form->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);

    auto parseColor3Value = [](const QString& value, float& r, float& g, float& b) {
        const QStringList parts = value.split(QRegularExpression("[,\\s]+"), Qt::SkipEmptyParts);
        if (parts.size() < 3) return false;
        bool ok1 = false, ok2 = false, ok3 = false;
        r = parts[0].toFloat(&ok1);
        g = parts[1].toFloat(&ok2);
        b = parts[2].toFloat(&ok3);
        return ok1 && ok2 && ok3;
    };
    auto formatColor3 = [](float r, float g, float b) {
        return QString("%1, %2, %3").arg(r, 0, 'g', 4).arg(g, 0, 'g', 4).arg(b, 0, 'g', 4);
    };

    for (const MaterialXInputParam& input : materialX_.inputs) {
        if (input.name.isEmpty()) continue;

        if (!input.nodename.isEmpty()) {
            auto* linked = new QLabel("← " + input.nodename);
            linked->setStyleSheet("color: #8eb7ff;");
            linked->setToolTip("Connected input (disconnect the wire to edit a constant value)");
            form->addRow(input.name, linked);
            continue;
        }

        const QString type = input.type.toLower();
        const QString inputName = input.name;
        auto commit = [this, inputName](const QString& value) {
            if (updating_ || !materialXMode_ || !materialX_.hostMaterial) return;
            emit materialXInputEdited(materialX_.hostMaterial, materialX_.name, inputName, value);
        };

        if (type == "filename") {
            auto* row = new QWidget();
            auto* rowLayout = new QHBoxLayout(row);
            rowLayout->setContentsMargins(0, 0, 0, 0);
            auto* edit = new QLineEdit(input.value);
            auto* browse = new QPushButton("…");
            browse->setFixedWidth(28);
            connect(edit, &QLineEdit::editingFinished, this, [edit, commit] { commit(edit->text()); });
            connect(browse, &QPushButton::clicked, this, [this, edit, commit] {
                const QString path = QFileDialog::getOpenFileName(
                    this, "Choose file", edit->text(),
                    "Images (*.png *.jpg *.jpeg *.exr *.hdr *.tif *.tiff *.bmp *.webp);;All Files (*)");
                if (path.isEmpty()) return;
                edit->setText(path);
                commit(path);
            });
            rowLayout->addWidget(edit, 1);
            rowLayout->addWidget(browse);
            form->addRow(input.name, row);
            continue;
        }

        if (type == "boolean" || type == "bool") {
            auto* box = new QCheckBox();
            box->setChecked(input.value == "true" || input.value == "1");
            connect(box, &QCheckBox::toggled, this,
                    [commit](bool checked) { commit(checked ? "true" : "false"); });
            form->addRow(input.name, box);
            continue;
        }

        if (type == "float" || type == "integer" || type == "int") {
            auto* spin = makeDoubleSpin(input.value.toDouble(), -1.0e6, 1.0e6);
            spin->setDecimals(type.startsWith("int") ? 0 : 4);
            const QString inputType = input.type;
            connect(spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                    [commit, inputType](double value) {
                        commit(inputType.startsWith("int") ? QString::number(int(value))
                                                           : QString::number(value, 'g', 6));
                    });
            form->addRow(input.name, spin);
            continue;
        }

        if (type == "color3" || type == "color4" || type == "vector3") {
            float r = 1, g = 1, b = 1;
            parseColor3Value(input.value, r, g, b);
            auto* row = new QWidget();
            auto* rowLayout = new QHBoxLayout(row);
            rowLayout->setContentsMargins(0, 0, 0, 0);
            auto* edit = new QLineEdit(input.value);
            auto* swatch = new QPushButton();
            swatch->setFixedSize(28, 22);
            auto updateSwatch = [swatch](float rr, float gg, float bb) {
                swatch->setStyleSheet(QString("background:%1; border:1px solid #222;")
                                          .arg(QColor::fromRgbF(qBound(0.0, double(rr), 1.0),
                                                                qBound(0.0, double(gg), 1.0),
                                                                qBound(0.0, double(bb), 1.0))
                                                   .name()));
            };
            updateSwatch(r, g, b);
            connect(edit, &QLineEdit::editingFinished, this, [edit, commit, updateSwatch, parseColor3Value] {
                float rr = 1, gg = 1, bb = 1;
                parseColor3Value(edit->text(), rr, gg, bb);
                updateSwatch(rr, gg, bb);
                commit(edit->text().trimmed());
            });
            connect(swatch, &QPushButton::clicked, this,
                    [this, edit, commit, updateSwatch, parseColor3Value, formatColor3, inputName] {
                        float rr = 1, gg = 1, bb = 1;
                        parseColor3Value(edit->text(), rr, gg, bb);
                        const QColor chosen = QColorDialog::getColor(
                            QColor::fromRgbF(qBound(0.0, double(rr), 1.0), qBound(0.0, double(gg), 1.0),
                                             qBound(0.0, double(bb), 1.0)),
                            this, "Pick " + inputName);
                        if (!chosen.isValid()) return;
                        const QString value =
                            formatColor3(float(chosen.redF()), float(chosen.greenF()), float(chosen.blueF()));
                        edit->setText(value);
                        updateSwatch(float(chosen.redF()), float(chosen.greenF()), float(chosen.blueF()));
                        commit(value);
                    });
            rowLayout->addWidget(edit, 1);
            rowLayout->addWidget(swatch);
            form->addRow(input.name, row);
            continue;
        }

        auto* edit = new QLineEdit(input.value);
        connect(edit, &QLineEdit::editingFinished, this, [edit, commit] { commit(edit->text().trimmed()); });
        form->addRow(input.name, edit);
    }

    contentLayout_->addWidget(paramsBox);
    contentLayout_->addStretch(1);
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
                slider = new NoWheelSlider(Qt::Horizontal);
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
            auto* spin = new NoWheelSpinBox();
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
            auto* combo = new NoWheelComboBox();
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
