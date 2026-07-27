#include "ui/parameter_panel.h"

#include <QAbstractSpinBox>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMimeData>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QVector3D>
#include <algorithm>
#include <cmath>
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

double defaultSpinStep(double minimum, double maximum) {
    const double span = maximum - minimum;
    if (span <= 2.0) return 0.01;
    if (span <= 20.0) return 0.1;
    return 1.0;
}

NoWheelDoubleSpinBox* makeDoubleSpin(double value, double minimum, double maximum) {
    auto* spin = new NoWheelDoubleSpinBox();
    spin->setRange(minimum, maximum);
    spin->setDecimals(6);
    spin->setSingleStep(defaultSpinStep(minimum, maximum));
    spin->setButtonSymbols(QAbstractSpinBox::NoButtons);
    spin->setValue(value);
    spin->setKeyboardTracking(false);
    spin->setMinimumWidth(64);
    spin->setMaximumWidth(96);
    return spin;
}

// Line-edit + soft slider: type freely (caret anywhere); slider only authors within [lo,hi].
QWidget* makeFreeFloatSliderRow(double value, double sliderMin, double sliderMax,
                                const std::function<void(double)>& onCommit) {
    auto* container = new QWidget();
    auto* layout = new QHBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);

    auto* edit = new FreeFloatLineEdit(value);
    edit->setMinimumWidth(64);
    edit->setMaximumWidth(96);
    layout->addWidget(edit, 0);

    auto* slider = new NoWheelSlider(Qt::Horizontal);
    slider->setRange(0, 1000);
    const double span = std::max(1e-9, sliderMax - sliderMin);
    const double t = (value - sliderMin) / span;
    slider->setValue(int(std::lround(std::clamp(t, 0.0, 1.0) * 1000.0)));
    layout->addWidget(slider, 1);

    auto syncSlider = [slider, sliderMin, span](double v) {
        QSignalBlocker blocker(slider);
        slider->setValue(int(std::lround(std::clamp((v - sliderMin) / span, 0.0, 1.0) * 1000.0)));
    };

    QObject::connect(edit, &QLineEdit::editingFinished, container, [edit, onCommit, syncSlider] {
        bool ok = false;
        const double v = edit->value(&ok);
        if (!ok) {
            edit->setValue(0.0, true);
            return;
        }
        edit->setValue(v, true);  // normalize formatting after commit
        syncSlider(v);
        onCommit(v);
    });
    QObject::connect(slider, &QSlider::valueChanged, container, [edit, sliderMin, span, onCommit](int pos) {
        const double v = sliderMin + span * (double(pos) / 1000.0);
        edit->setValue(v, true);
        onCommit(v);
    });
    return container;
}

QString prettyMaterialXLabel(const QString& name) {
    static const QHash<QString, QString> special = {
        {QStringLiteral("specular_IOR"), QStringLiteral("Specular IOR")},
        {QStringLiteral("coat_IOR"), QStringLiteral("Coat IOR")},
        {QStringLiteral("thin_film_IOR"), QStringLiteral("Thin Film IOR")},
        {QStringLiteral("subsurface_scale"), QStringLiteral("Subsurface Scale")},
    };
    if (special.contains(name)) return special.value(name);

    QStringList parts = name.split(QLatin1Char('_'), Qt::SkipEmptyParts);
    for (QString& part : parts) {
        if (part.compare(QLatin1String("ior"), Qt::CaseInsensitive) == 0) {
            part = QStringLiteral("IOR");
        } else if (part.compare(QLatin1String("rgb"), Qt::CaseInsensitive) == 0) {
            part = QStringLiteral("RGB");
        } else if (part.compare(QLatin1String("uv"), Qt::CaseInsensitive) == 0) {
            part = QStringLiteral("UV");
        } else if (!part.isEmpty()) {
            part = part.left(1).toUpper() + part.mid(1).toLower();
        }
    }
    return parts.join(QLatin1Char(' '));
}

// Arnold / Autodesk Standard Surface style UI ranges for MaterialX floats.
void materialXFloatRange(const QString& name, double& lo, double& hi) {
    const QString n = name.toLower();
    if (n.endsWith(QLatin1String("_ior")) || n == QLatin1String("ior")) {
        lo = 0.0;
        hi = 5.0;
        return;
    }
    if (n == QLatin1String("subsurface_scale")) {
        // Arnold Scale in scene units (metres): MFP = Scale * Radius. Soft slider 0..10.
        lo = 0.0;
        hi = 10.0;
        return;
    }
    if (n == QLatin1String("scale")) {
        lo = 0.0;
        hi = 10.0;
        return;
    }
    if (n == QLatin1String("emission") || n.contains(QLatin1String("intensity")) ||
        n.contains(QLatin1String("exposure"))) {
        lo = 0.0;
        hi = 10.0;
        return;
    }
    if (n.contains(QLatin1String("anisotropy")) && n.contains(QLatin1String("subsurface"))) {
        lo = -1.0;
        hi = 1.0;
        return;
    }
    if (n.contains(QLatin1String("metalness")) || n.contains(QLatin1String("roughness")) ||
        n == QLatin1String("specular") || n == QLatin1String("base") || n == QLatin1String("transmission") ||
        n == QLatin1String("subsurface") || n == QLatin1String("sheen") || n == QLatin1String("coat") ||
        n == QLatin1String("mix") || n.contains(QLatin1String("opacity")) ||
        n.contains(QLatin1String("weight"))) {
        lo = 0.0;
        hi = 1.0;
        return;
    }
    if (n.contains(QLatin1String("rotation")) || n.contains(QLatin1String("anisotropy"))) {
        lo = 0.0;
        hi = 1.0;
        return;
    }
    // Generic MaterialX floats — keep a usable authoring range (not ±1e6).
    lo = -10.0;
    hi = 10.0;
}

bool parseFloatList(const QString& value, QVector<double>& out, int expectedMin) {
    out.clear();
    const QStringList parts = value.split(QRegularExpression("[,\\s]+"), Qt::SkipEmptyParts);
    for (const QString& part : parts) {
        bool ok = false;
        const double v = part.toDouble(&ok);
        if (!ok) return false;
        out.push_back(v);
    }
    return out.size() >= expectedMin;
}

QString formatFloatList(const QVector<double>& values) {
    QStringList parts;
    for (double v : values) parts << QString::number(v, 'g', 6);
    return parts.join(QStringLiteral(", "));
}

NoWheelSpinBox* makeIntSpin(int value, int minimum, int maximum) {
    auto* spin = new NoWheelSpinBox();
    spin->setRange(minimum, maximum);
    spin->setSingleStep(1);
    spin->setButtonSymbols(QAbstractSpinBox::NoButtons);
    spin->setValue(value);
    spin->setKeyboardTracking(false);
    spin->setMinimumWidth(72);
    spin->setMaximumWidth(96);
    return spin;
}

QWidget* makeSpinSliderRow(QWidget* spin, double value, double minimum, double maximum,
                           const std::function<void(double)>& onSpinChanged) {
    auto* container = new QWidget();
    auto* layout = new QHBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);
    layout->addWidget(spin, 0);

    auto* slider = new NoWheelSlider(Qt::Horizontal);
    slider->setRange(0, 1000);
    const double span = std::max(1e-9, maximum - minimum);
    const double t = (value - minimum) / span;
    slider->setValue(int(std::lround(std::clamp(t, 0.0, 1.0) * 1000.0)));
    layout->addWidget(slider, 1);

    if (auto* doubleSpin = qobject_cast<QDoubleSpinBox*>(spin)) {
        QObject::connect(doubleSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), container,
                         [slider, minimum, span, onSpinChanged](double v) {
                             QSignalBlocker blocker(slider);
                             slider->setValue(int(std::lround(std::clamp((v - minimum) / span, 0.0, 1.0) * 1000.0)));
                             onSpinChanged(v);
                         });
        QObject::connect(slider, &QSlider::valueChanged, container, [doubleSpin, minimum, span](int pos) {
            const double v = minimum + span * (double(pos) / 1000.0);
            doubleSpin->setValue(v);
        });
    } else if (auto* intSpin = qobject_cast<QSpinBox*>(spin)) {
        QObject::connect(intSpin, QOverload<int>::of(&QSpinBox::valueChanged), container,
                         [slider, minimum, span, onSpinChanged](int v) {
                             QSignalBlocker blocker(slider);
                             slider->setValue(int(std::lround(std::clamp((double(v) - minimum) / span, 0.0, 1.0) * 1000.0)));
                             onSpinChanged(double(v));
                         });
        QObject::connect(slider, &QSlider::valueChanged, container, [intSpin, minimum, span](int pos) {
            const double v = minimum + span * (double(pos) / 1000.0);
            intSpin->setValue(int(std::lround(v)));
        });
    }
    return container;
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
    focusPickButton_ = nullptr;
    rebuild();
}

void ParameterPanel::setNode(Node* node) {
    if (!materialXMode_ && node_ == node) return;
    node_ = node;
    materialXMode_ = false;
    materialX_ = {};
    focusPickButton_ = nullptr;
    rebuild();
}

void ParameterPanel::setMaterialXSelection(const MaterialXSelection& selection) {
    node_ = selection.hostMaterial;
    materialXMode_ = selection.hostMaterial && !selection.name.isEmpty();
    materialX_ = selection;
    focusPickButton_ = nullptr;
    if (!materialXMode_) {
        materialX_ = {};
        rebuild();
        return;
    }
    rebuild();
}

void ParameterPanel::refresh() { rebuild(); }

void ParameterPanel::setFocusPickActive(bool active) {
    focusPickActive_ = active;
    if (focusPickButton_) {
        const QSignalBlocker blocker(focusPickButton_);
        focusPickButton_->setChecked(active);
    }
}

void ParameterPanel::rebuild() {
    updating_ = true;
    focusPickButton_ = nullptr;
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
        auto* hint = new QLabel("No node selected.\n\nPress Tab in the Scene Network to add one,\n"
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

    auto makeComponentRow = [this](int count, const QVector<double>& values, double lo, double hi, bool showSwatch,
                                   const QString& pickerTitle, const std::function<void(const QString&)>& commit) {
        auto* row = new QWidget();
        auto* layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(4);

        QVector<QDoubleSpinBox*> spins;
        spins.reserve(count);
        for (int i = 0; i < count; ++i) {
            const double v = i < values.size() ? values[i] : (i < 3 ? 0.0 : 1.0);
            auto* spin = makeDoubleSpin(std::clamp(v, lo, hi), lo, hi);
            spin->setDecimals(3);
            spin->setMaximumWidth(64);
            spins.push_back(spin);
            layout->addWidget(spin, 0);
        }

        auto pushValue = [spins, commit] {
            QVector<double> out;
            out.reserve(spins.size());
            for (QDoubleSpinBox* spin : spins) out.push_back(spin->value());
            commit(formatFloatList(out));
        };

        QPushButton* swatch = nullptr;
        if (showSwatch) {
            swatch = new QPushButton();
            swatch->setFixedSize(28, 22);
            swatch->setToolTip("Pick colour");
            // Swatch on the right of the component fields (Houdini-like, mirrored).
            layout->addWidget(swatch, 0);
        }
        layout->addStretch(1);

        auto refreshSwatch = [spins, swatch] {
            if (!swatch || spins.size() < 3) return;
            swatch->setStyleSheet(
                QString("background:%1; border:1px solid #222;")
                    .arg(QColor::fromRgbF(qBound(0.0, spins[0]->value(), 1.0), qBound(0.0, spins[1]->value(), 1.0),
                                          qBound(0.0, spins[2]->value(), 1.0))
                             .name()));
        };
        refreshSwatch();

        if (swatch) {
            connect(swatch, &QPushButton::clicked, this, [this, spins, pushValue, refreshSwatch, pickerTitle] {
                if (spins.size() < 3) return;
                const QColor chosen = QColorDialog::getColor(
                    QColor::fromRgbF(qBound(0.0, spins[0]->value(), 1.0), qBound(0.0, spins[1]->value(), 1.0),
                                     qBound(0.0, spins[2]->value(), 1.0)),
                    this, pickerTitle);
                if (!chosen.isValid()) return;
                QSignalBlocker b0(spins[0]);
                QSignalBlocker b1(spins[1]);
                QSignalBlocker b2(spins[2]);
                spins[0]->setValue(chosen.redF());
                spins[1]->setValue(chosen.greenF());
                spins[2]->setValue(chosen.blueF());
                refreshSwatch();
                pushValue();
            });
        }

        for (QDoubleSpinBox* spin : spins) {
            connect(spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                    [pushValue, refreshSwatch](double) {
                        refreshSwatch();
                        pushValue();
                    });
        }
        return row;
    };

    for (const MaterialXInputParam& input : materialX_.inputs) {
        if (input.name.isEmpty()) continue;
        const QString label = prettyMaterialXLabel(input.name);

        if (!input.nodename.isEmpty()) {
            auto* linked = new QLabel("← " + input.nodename);
            linked->setStyleSheet("color: #8eb7ff;");
            linked->setToolTip("Connected input (disconnect the wire to edit a constant value)");
            form->addRow(label, linked);
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
            form->addRow(label, row);
            continue;
        }

        if (type == "boolean" || type == "bool") {
            auto* box = new QCheckBox();
            box->setChecked(input.value == "true" || input.value == "1");
            connect(box, &QCheckBox::toggled, this,
                    [commit](bool checked) { commit(checked ? "true" : "false"); });
            form->addRow(label, box);
            continue;
        }

        if (type == "float" || type == "integer" || type == "int") {
            const bool isInt = type.startsWith("int");
            double lo = -10.0;
            double hi = 10.0;
            if (isInt) {
                lo = -1000.0;
                hi = 1000.0;
            } else {
                materialXFloatRange(input.name, lo, hi);
            }
            double value = input.value.toDouble();
            if (!std::isfinite(value)) value = lo;
            // Soft slider range only — typed values may go beyond (esp. subsurface_scale).
            const QString inputType = input.type;
            const QString tip =
                (input.name == QLatin1String("subsurface_scale"))
                    ? QStringLiteral("Arnold Scale in scene units (metres). "
                                     "Mean free path = Scale × Radius. 1 = 1 metre.")
                    : QString();
            QWidget* row = makeFreeFloatSliderRow(value, lo, hi, [commit, inputType, isInt](double v) {
                if (isInt) commit(QString::number(int(std::lround(v))));
                else commit(QString::number(v, 'g', 9));
            });
            if (!tip.isEmpty()) row->setToolTip(tip);
            form->addRow(label, row);
            continue;
        }

        if (type == "color3" || type == "color4" || type.startsWith("vector") || type.startsWith("matrix")) {
            int count = 3;
            if (type == "color4" || type == "vector4") count = 4;
            else if (type == "vector2") count = 2;
            else if (type == "matrix33") count = 9;
            else if (type == "matrix44") count = 16;
            else if (type.startsWith("vector") && type.size() > 6) {
                bool ok = false;
                const int n = type.mid(6).toInt(&ok);
                if (ok && n > 0) count = n;
            }

            QVector<double> values;
            if (!parseFloatList(input.value, values, 1)) {
                values = QVector<double>(count, type.startsWith("color") ? 1.0 : 0.0);
            }
            while (values.size() < count) values.push_back(type.startsWith("color") ? 1.0 : 0.0);

            const bool isColor = type.startsWith("color");
            const double lo = isColor ? 0.0 : -10.0;
            const double hi = isColor ? 1.0 : 10.0;
            // subsurface_radius: Arnold RGB MFP weights (often 0..1 relative; R>G>B for skin).
            const double compLo = (input.name == QLatin1String("subsurface_radius")) ? 0.0 : lo;
            const double compHi = (input.name == QLatin1String("subsurface_radius")) ? 1.0 : hi;

            if (count <= 4) {
                QWidget* row = makeComponentRow(count, values, compLo, compHi, isColor, "Pick " + label, commit);
                if (input.name == QLatin1String("subsurface_radius")) {
                    row->setToolTip(
                        "Arnold RGB mean free path weights. MFP = Scale × Radius per channel.\n"
                        "Higher red than green/blue (e.g. 1, 0.35, 0.2) gives warm edge fringing.");
                }
                form->addRow(label, row);
            } else {
                // Matrices: compact grid of component fields.
                auto* row = new QWidget();
                auto* grid = new QGridLayout(row);
                grid->setContentsMargins(0, 0, 0, 0);
                grid->setHorizontalSpacing(3);
                grid->setVerticalSpacing(3);
                const int dim = (count == 9) ? 3 : 4;
                QVector<QDoubleSpinBox*> spins;
                spins.reserve(count);
                for (int i = 0; i < count; ++i) {
                    auto* spin = makeDoubleSpin(std::clamp(values[i], -10.0, 10.0), -10.0, 10.0);
                    spin->setDecimals(3);
                    spin->setMaximumWidth(58);
                    spins.push_back(spin);
                    grid->addWidget(spin, i / dim, i % dim);
                }
                auto pushValue = [spins, commit] {
                    QVector<double> out;
                    for (QDoubleSpinBox* spin : spins) out.push_back(spin->value());
                    commit(formatFloatList(out));
                };
                for (QDoubleSpinBox* spin : spins) {
                    connect(spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                            [pushValue](double) { pushValue(); });
                }
                form->addRow(label, row);
            }
            continue;
        }

        auto* edit = new QLineEdit(input.value);
        connect(edit, &QLineEdit::editingFinished, this, [edit, commit] { commit(edit->text().trimmed()); });
        form->addRow(label, edit);
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
            const double lo = parameter.minValue;
            const double hi = parameter.maxValue;
            QWidget* slider = makeFreeFloatSliderRow(parameter.toDouble(), lo, hi,
                                                    [notify](double value) { notify(value); });
            // Camera DOF: Focus Pick next to Focus Distance.
            if (name == QLatin1String("focusdistance") && node_ && node_->typeName() == QLatin1String("camera")) {
                auto* row = new QWidget();
                auto* layout = new QHBoxLayout(row);
                layout->setContentsMargins(0, 0, 0, 0);
                layout->setSpacing(6);
                layout->addWidget(slider, 1);
                auto* pick = new QPushButton("Focus Pick");
                pick->setCheckable(true);
                pick->setChecked(focusPickActive_);
                pick->setToolTip("Click in the viewport on geometry to set focus distance for DOF");
                pick->setMaximumWidth(96);
                focusPickButton_ = pick;
                connect(pick, &QPushButton::toggled, this, [this](bool on) {
                    focusPickActive_ = on;
                    emit focusPickToggled(on);
                });
                layout->addWidget(pick, 0);
                return row;
            }
            // Camera DOF: common f-stop presets next to F-Stop.
            if (name == QLatin1String("fstop") && node_ && node_->typeName() == QLatin1String("camera")) {
                auto* row = new QWidget();
                auto* layout = new QHBoxLayout(row);
                layout->setContentsMargins(0, 0, 0, 0);
                layout->setSpacing(6);
                layout->addWidget(slider, 1);

                static const double kFStopPresets[] = {1.0,  1.2, 1.4, 1.8, 2.0,  2.5,  2.8, 4.0,
                                                      5.6,  8.0, 11.0, 16.0, 22.0, 32.0, 64.0};
                auto* combo = new QComboBox();
                combo->setToolTip("F-Stop presets");
                combo->setMaximumWidth(78);
                combo->addItem(QStringLiteral("f/…"));
                const double current = parameter.toDouble();
                int matched = 0;
                for (double stop : kFStopPresets) {
                    QString label;
                    if (std::fabs(stop - std::round(stop)) < 1e-6)
                        label = QString::number(int(std::lround(stop)));
                    else
                        label = QString::number(stop, 'g', 3);
                    combo->addItem(QStringLiteral("f/%1").arg(label), stop);
                    if (matched == 0 && std::fabs(current - stop) < 1e-4) matched = combo->count() - 1;
                }
                combo->setCurrentIndex(matched);
                connect(combo, QOverload<int>::of(&QComboBox::activated), this,
                        [this, notify, combo](int index) {
                            if (updating_ || index <= 0) return;
                            notify(combo->itemData(index).toDouble());
                            // Rebuild so the free-float slider reflects the preset.
                            QMetaObject::invokeMethod(this, [this] { refresh(); }, Qt::QueuedConnection);
                        });
                layout->addWidget(combo, 0);
                return row;
            }
            return slider;
        }
        case ParamType::Int: {
            const int lo = int(parameter.minValue);
            const int hi = int(parameter.maxValue);
            auto* spin = makeIntSpin(parameter.toInt(), lo, hi);
            return makeSpinSliderRow(spin, double(parameter.toInt()), double(lo), double(hi),
                                     [notify](double value) { notify(int(std::lround(value))); });
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
                spins[i]->setMaximumWidth(80);
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
