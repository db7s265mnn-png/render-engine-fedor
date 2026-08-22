#include "ui/parameter_panel.h"

#include <QAbstractButton>
#include <QAbstractSpinBox>
#include <QButtonGroup>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QLayout>
#include <QLayoutItem>
#include <QLineEdit>
#include <QList>
#include <QMimeData>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QVector3D>
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "nodes/node_registry.h"
#include "core/expr_eval.h"
#include "io/tx_convert.h"
#include "render/pixel_filter.h"
#include "ui/numeric_editors.h"
#include "ui/texture_file_dialog.h"
#include "ui/theme.h"

namespace sol {
namespace {

// Wrapping Houdini-style folder strip (QTabWidget cannot wrap in a 280px dock,
// and Fusion + pane{top:-1px} can cover the tab bar entirely).
class FlowLayout final : public QLayout {
public:
    explicit FlowLayout(QWidget* parent = nullptr, int hSpacing = 4, int vSpacing = 4)
        : QLayout(parent), hSpacing_(hSpacing), vSpacing_(vSpacing) {}
    ~FlowLayout() override {
        while (QLayoutItem* item = takeAt(0)) delete item;
    }

    void addItem(QLayoutItem* item) override { items_.append(item); }
    int count() const override { return items_.size(); }
    QLayoutItem* itemAt(int index) const override { return items_.value(index); }
    QLayoutItem* takeAt(int index) override {
        if (index < 0 || index >= items_.size()) return nullptr;
        return items_.takeAt(index);
    }
    Qt::Orientations expandingDirections() const override { return {}; }
    bool hasHeightForWidth() const override { return true; }
    int heightForWidth(int width) const override { return doLayout(QRect(0, 0, width, 0), true); }
    void setGeometry(const QRect& rect) override {
        QLayout::setGeometry(rect);
        doLayout(rect, false);
    }
    QSize sizeHint() const override { return minimumSize(); }
    QSize minimumSize() const override {
        QSize size;
        for (QLayoutItem* item : items_) size = size.expandedTo(item->minimumSize());
        const QMargins m = contentsMargins();
        size += QSize(m.left() + m.right(), m.top() + m.bottom());
        return size;
    }

private:
    int doLayout(const QRect& rect, bool testOnly) const {
        int x = rect.x();
        int y = rect.y();
        int lineHeight = 0;
        for (QLayoutItem* item : items_) {
            const QSize hint = item->sizeHint();
            if (x + hint.width() > rect.right() + 1 && lineHeight > 0) {
                x = rect.x();
                y += lineHeight + vSpacing_;
                lineHeight = 0;
            }
            if (!testOnly) item->setGeometry(QRect(QPoint(x, y), hint));
            x += hint.width() + hSpacing_;
            lineHeight = qMax(lineHeight, hint.height());
        }
        return y + lineHeight - rect.y();
    }

    QList<QLayoutItem*> items_;
    int hSpacing_ = 4;
    int vSpacing_ = 4;
};

QComboBox* makeColorSpaceCombo(const QString& current, const std::function<void(const QString&)>& commit) {
    auto* combo = new QComboBox();
    combo->setEditable(true);
    combo->blockSignals(true);
    QStringList curated;
    for (const std::string& name : txCuratedColorSpaces()) curated << QString::fromStdString(name);
    combo->addItems(curated);
    const int idx = combo->findText(current);
    if (idx >= 0) combo->setCurrentIndex(idx);
    else combo->setEditText(current.isEmpty() ? QStringLiteral("auto") : current);
    combo->blockSignals(false);
    combo->setToolTip(QStringLiteral(
        "Arnold-style input colour space. Cook converts textures to ACEScg.\n"
        "auto: HDR/EXR \u2192 Linear sRGB, 8-bit \u2192 sRGB Texture, data \u2192 Raw.\n"
        "ACEScg / Raw: no convert. Linear sRGB: Rec.709 primaries \u2192 AP1."));
    QObject::connect(combo, &QComboBox::textActivated, combo, [commit](const QString& text) {
        if (!text.isEmpty()) commit(text);
    });
    if (QLineEdit* edit = combo->lineEdit()) {
        QObject::connect(edit, &QLineEdit::editingFinished, combo, [combo, commit] {
            const QString text = combo->currentText();
            if (!text.isEmpty()) commit(text);
        });
    }
    return combo;
}

void applyExpressionFieldStyle(QWidget* widget, bool isExpression) {
    if (!widget) return;
    widget->setStyleSheet(isExpression ? expressionFieldStyleSheet() : normalFieldStyleSheet());
}

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

// Line-edit + soft slider. Edit accepts Houdini-style expressions ($F, math).
QWidget* makeFreeFloatSliderRow(double value, const QString& expression, double sliderMin,
                                double sliderMax, const std::function<void(const QString&)>& onCommitText) {
    auto* container = new QWidget();
    auto* layout = new QHBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);

    auto* edit = new FreeFloatLineEdit(value);
    edit->setMinimumWidth(64);
    edit->setMaximumWidth(120);
    if (!expression.isEmpty()) {
        edit->setText(expression);
        applyExpressionFieldStyle(edit, true);
    }
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

    QObject::connect(edit, &QLineEdit::editingFinished, container, [edit, onCommitText, syncSlider] {
        const QString text = edit->text().trimmed();
        applyExpressionFieldStyle(edit, looksLikeExpression(text));
        if (looksLikeExpression(text)) {
            double v = 0.0;
            if (evalExpression(text, exprFrame(), v)) syncSlider(v);
            onCommitText(text);
            return;
        }
        bool ok = false;
        const double v = QLocale::c().toDouble(text, &ok);
        if (!ok) {
            edit->setValue(0.0, true);
            onCommitText(QStringLiteral("0"));
            return;
        }
        edit->setValue(v, true);
        syncSlider(v);
        onCommitText(QString::number(v, 'g', 9));
    });
    QObject::connect(slider, &QSlider::valueChanged, container,
                     [edit, sliderMin, span, onCommitText](int pos) {
                         const double v = sliderMin + span * (double(pos) / 1000.0);
                         edit->setValue(v, true);
                         applyExpressionFieldStyle(edit, false);
                         onCommitText(QString::number(v, 'g', 9));
                     });
    return container;
}

// Back-compat wrapper used by older call sites expecting double callback.
QWidget* makeFreeFloatSliderRow(double value, double sliderMin, double sliderMax,
                                const std::function<void(double)>& onCommit) {
    return makeFreeFloatSliderRow(value, QString(), sliderMin, sliderMax,
                                  [onCommit](const QString& text) {
                                      bool ok = false;
                                      double v = 0.0;
                                      if (looksLikeExpression(text))
                                          evalExpression(text, exprFrame(), v);
                                      else
                                          v = QLocale::c().toDouble(text, &ok);
                                      onCommit(v);
                                  });
}

QString prettyMaterialXLabel(const QString& name) {
    static const QHash<QString, QString> special = {
        {QStringLiteral("specular_IOR"), QStringLiteral("Specular IOR")},
        {QStringLiteral("coat_IOR"), QStringLiteral("Coat IOR")},
        {QStringLiteral("thin_film_IOR"), QStringLiteral("Thin Film IOR")},
        {QStringLiteral("internal_reflections"), QStringLiteral("Internal Reflections")},
        {QStringLiteral("subsurface_scale"), QStringLiteral("Subsurface Scale")},
        {QStringLiteral("input_per_axis"), QStringLiteral("Input Per Axis")},
        {QStringLiteral("file"), QStringLiteral("Input")},
        {QStringLiteral("filex"), QStringLiteral("Input X")},
        {QStringLiteral("filey"), QStringLiteral("Input Y")},
        {QStringLiteral("filez"), QStringLiteral("Input Z")},
        {QStringLiteral("conductor_eta"), QStringLiteral("Conductor η")},
        {QStringLiteral("conductor_k"), QStringLiteral("Conductor κ")},
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
    if (n == QLatin1String("dispersion_abbe") || n == QLatin1String("abbe") ||
        n.endsWith(QLatin1String("_abbe"))) {
        // Arnold-style Abbe Vd: 0 = off (no dispersion). Negatives are invalid.
        lo = 0.0;
        hi = 200.0;
        return;
    }
    if (n == QLatin1String("blend")) {
        lo = 0.0;
        hi = 1.0;
        return;
    }
    if (n == QLatin1String("rotate")) {
        lo = 0.0;
        hi = 360.0;
        return;
    }
    if (n == QLatin1String("scale")) {
        lo = 0.0;
        hi = 10.0;
        return;
    }
    // Soft slider only — typed values may exceed these for procedural authoring.
    if (n.contains(QLatin1String("freq")) || n.contains(QLatin1String("frequency")) ||
        n == QLatin1String("lacunarity")) {
        lo = 0.0;
        hi = 100.0;
        return;
    }
    if (n.contains(QLatin1String("amplitude")) || n.contains(QLatin1String("offset")) ||
        n == QLatin1String("pivot")) {
        lo = -10.0;
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
        n.contains(QLatin1String("weight")) || n == QLatin1String("shadow_opacity")) {
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
    setMinimumWidth(280);
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);

    // Folder tabs own their own scroll areas so the tab bar stays pinned
    // (Houdini-style). MaterialX pages wrap themselves the same way.
    content_ = new QWidget();
    content_->setMinimumWidth(0);
    content_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    contentLayout_ = new QVBoxLayout(content_);
    contentLayout_->setContentsMargins(8, 8, 8, 8);
    contentLayout_->setSpacing(8);
    outer->addWidget(content_);

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
    // Tear down editors with signals blocked. QComboBox emits
    // currentIndexChanged(-1) on destroy; without a block that writes -1 into
    // integrator / causticsengine and silently kills MNEE/Photon/BDPT.
    QLayoutItem* child = nullptr;
    while ((child = contentLayout_->takeAt(0)) != nullptr) {
        if (QWidget* w = child->widget()) {
            w->setUpdatesEnabled(false);
            const auto buttons = w->findChildren<QComboBox*>();
            for (QComboBox* combo : buttons) combo->blockSignals(true);
            w->deleteLater();
        }
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
    // Header: node name plus type description (stays above the folder tabs).
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

    // Houdini-style wrapping folder buttons. Only the active folder's parameters
    // are shown. Orange = selected.
    QStringList groups;
    groups << QString();
    for (const Parameter& parameter : node_->parameters()) {
        if (parameter.name == "mtlx") continue;
        if (!groups.contains(parameter.group)) groups << parameter.group;
    }

    auto makeFolderPage = [](QFormLayout** formOut) -> QWidget* {
        auto* scroll = new QScrollArea();
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        auto* inner = new QWidget();
        auto* layout = new QVBoxLayout(inner);
        layout->setContentsMargins(6, 8, 6, 8);
        layout->setSpacing(4);
        auto* form = new QFormLayout();
        form->setContentsMargins(0, 0, 0, 0);
        form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        form->setRowWrapPolicy(QFormLayout::WrapLongRows);
        layout->addLayout(form);
        layout->addStretch(1);
        scroll->setWidget(inner);
        *formOut = form;
        return scroll;
    };

    struct FolderPage {
        QString name;
        QWidget* page = nullptr;
        QFormLayout* form = nullptr;
    };
    std::vector<FolderPage> pages;

    for (const QString& group : groups) {
        FolderPage* folder = nullptr;
        for (Parameter& parameter : node_->parameters()) {
            if (parameter.name == "mtlx") continue;
            if (parameter.name.startsWith(QLatin1String("_"))) continue;
            if (parameter.group != group) continue;
            if (!evaluateVisibleWhen(parameter.visibleWhen, *node_)) continue;
            if (!folder) {
                pages.emplace_back();
                folder = &pages.back();
                folder->name = group;
                folder->page = makeFolderPage(&folder->form);
            }
            QWidget* editor = createEditor(parameter);
            if (!editor) continue;
            if (!parameter.tooltip.isEmpty()) editor->setToolTip(parameter.tooltip);
            // Buttons carry their own label text — avoid "Render: [Render]".
            if (parameter.type == ParamType::Button)
                folder->form->addRow(QString(), editor);
            else
                folder->form->addRow(parameter.label, editor);
        }
    }

    if (pages.empty()) {
        contentLayout_->addStretch(1);
        return;
    }

    auto* strip = new QWidget();
    strip->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    auto* flow = new FlowLayout(strip, 4, 4);
    flow->setContentsMargins(0, 2, 0, 2);
    auto* buttons = new QButtonGroup(strip);
    buttons->setExclusive(true);
    auto* stack = new QStackedWidget();
    stack->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    const QString folderBtnCss = QStringLiteral(
        "QPushButton { background:#2a2e33; color:#d0d4da; border:1px solid #666b73;"
        " border-radius:2px; padding:4px 10px; min-height:22px; }"
        "QPushButton:hover { background:#3a3f46; color:#f0f2f5; }"
        "QPushButton:checked { background:#ffa82e; color:#1a1c20; font-weight:bold;"
        " border-color:#ffa82e; }");

    int restore = 0;
    const QString previous = lastFolderByType_.value(node_->typeName());
    for (int i = 0; i < static_cast<int>(pages.size()); ++i) {
        const FolderPage& folder = pages[static_cast<size_t>(i)];
        const QString title = folder.name.isEmpty() ? QStringLiteral("Parameters") : folder.name;
        auto* btn = new QPushButton(title);
        btn->setCheckable(true);
        btn->setCursor(Qt::PointingHandCursor);
        btn->setFocusPolicy(Qt::NoFocus);
        btn->setStyleSheet(folderBtnCss);
        flow->addWidget(btn);
        buttons->addButton(btn, i);
        stack->addWidget(folder.page);
        if (!previous.isEmpty() && title == previous) restore = i;
    }
    if (QAbstractButton* restored = buttons->button(restore)) restored->setChecked(true);
    stack->setCurrentIndex(restore);
    if (restore >= 0 && restore < stack->count())
        lastFolderByType_[node_->typeName()] = buttons->button(restore)->text();

    connect(buttons, &QButtonGroup::idClicked, this, [this, stack, buttons](int id) {
        if (id < 0 || id >= stack->count()) return;
        stack->setCurrentIndex(id);
        if (node_) lastFolderByType_[node_->typeName()] = buttons->button(id)->text();
    });

    contentLayout_->addWidget(strip);
    contentLayout_->addWidget(stack, 1);
}

void ParameterPanel::rebuildMaterialX() {
    auto* header = new QGroupBox("MaterialX Node");
    auto* headerLayout = new QFormLayout(header);

    if (materialX_.typeVariants.size() > 1) {
        auto* typeCombo = new QComboBox();
        for (const QString& variant : materialX_.typeVariants) typeCombo->addItem(variant);
        const int current = typeCombo->findText(materialX_.type);
        typeCombo->setCurrentIndex(current >= 0 ? current : 0);
        typeCombo->setToolTip("Output data type for this node");
        connect(typeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, typeCombo](int) {
            if (updating_ || !materialXMode_ || !materialX_.hostMaterial) return;
            const QString type = typeCombo->currentText();
            if (type.isEmpty() || type == materialX_.type) return;
            emit materialXTypeEdited(materialX_.hostMaterial, materialX_.name, type);
        });
        headerLayout->addRow("Type", typeCombo);
    } else {
        auto* typeLabel = new QLabel(materialX_.category +
                                     (materialX_.type.isEmpty() ? QString() : ("  ·  " + materialX_.type)));
        typeLabel->setStyleSheet("color: #969aa0;");
        typeLabel->setWordWrap(true);
        headerLayout->addRow("Type", typeLabel);
    }

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
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);

    auto makeComponentRow = [this](int count, const QVector<double>& values, double sliderLo, double sliderHi,
                                   bool showSwatch, const QString& pickerTitle,
                                   const std::function<void(const QString&)>& commit) {
        auto* row = new QWidget();
        auto* layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(4);

        // Soft slider range is display-only — typed values are never hard-clamped
        // (procedural freq / amplitude often need values well beyond the slider).
        QVector<FreeFloatLineEdit*> edits;
        edits.reserve(count);
        for (int i = 0; i < count; ++i) {
            const double v = i < values.size() ? values[i] : (i < 3 ? 0.0 : 1.0);
            auto* edit = new FreeFloatLineEdit(v);
            edit->setMinimumWidth(56);
            edit->setMaximumWidth(72);
            edits.push_back(edit);
            layout->addWidget(edit, 0);
        }

        auto pushValue = [edits, commit] {
            QVector<double> out;
            out.reserve(edits.size());
            for (FreeFloatLineEdit* edit : edits) {
                bool ok = false;
                double v = edit->value(&ok);
                if (!ok) v = 0.0;
                out.push_back(v);
            }
            commit(formatFloatList(out));
        };

        QPushButton* swatch = nullptr;
        if (showSwatch) {
            swatch = new QPushButton();
            swatch->setFixedSize(28, 22);
            swatch->setToolTip("Pick colour");
            layout->addWidget(swatch, 0);
        }
        layout->addStretch(1);

        auto refreshSwatch = [edits, swatch] {
            if (!swatch || edits.size() < 3) return;
            bool ok0 = false, ok1 = false, ok2 = false;
            const double r = edits[0]->value(&ok0);
            const double g = edits[1]->value(&ok1);
            const double b = edits[2]->value(&ok2);
            swatch->setStyleSheet(
                QString("background:%1; border:1px solid #222;")
                    .arg(QColor::fromRgbF(qBound(0.0, ok0 ? r : 0.0, 1.0), qBound(0.0, ok1 ? g : 0.0, 1.0),
                                          qBound(0.0, ok2 ? b : 0.0, 1.0))
                             .name()));
        };
        refreshSwatch();
        Q_UNUSED(sliderLo);
        Q_UNUSED(sliderHi);

        if (swatch) {
            connect(swatch, &QPushButton::clicked, this, [this, edits, pushValue, refreshSwatch, pickerTitle] {
                if (edits.size() < 3) return;
                bool ok0 = false, ok1 = false, ok2 = false;
                const QColor chosen = QColorDialog::getColor(
                    QColor::fromRgbF(qBound(0.0, edits[0]->value(&ok0), 1.0), qBound(0.0, edits[1]->value(&ok1), 1.0),
                                     qBound(0.0, edits[2]->value(&ok2), 1.0)),
                    this, pickerTitle);
                if (!chosen.isValid()) return;
                edits[0]->setValue(chosen.redF(), true);
                edits[1]->setValue(chosen.greenF(), true);
                edits[2]->setValue(chosen.blueF(), true);
                refreshSwatch();
                pushValue();
            });
        }

        for (FreeFloatLineEdit* edit : edits) {
            connect(edit, &QLineEdit::editingFinished, this, [edit, pushValue, refreshSwatch] {
                bool ok = false;
                const double v = edit->value(&ok);
                if (!ok) {
                    edit->setValue(0.0, true);
                    return;
                }
                edit->setValue(v, true);
                refreshSwatch();
                pushValue();
            });
        }
        return row;
    };

    for (const MaterialXInputParam& input : materialX_.inputs) {
        if (input.name.isEmpty()) continue;
        // Spectral-only conductor η/κ — hide unless PT Spectral is active.
        if ((input.name == QLatin1String("conductor_eta") || input.name == QLatin1String("conductor_k")) &&
            materialX_.activeIntegrator != 4) {
            continue;
        }
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
                const auto picked = TextureFileDialog::getOpenTexture(
                    this, QStringLiteral("Choose file"), edit->text(),
                    QStringLiteral(
                        "Images (*.png *.jpg *.jpeg *.exr *.hdr *.tx *.tif *.tiff *.bmp *.webp);;All Files (*)"));
                if (picked.path.isEmpty()) return;
                edit->setText(picked.path);
                applyExpressionFieldStyle(edit, looksLikeExpression(picked.path) ||
                                                    picked.path.contains(QLatin1Char('$')));
                commit(picked.path);
            });
            applyExpressionFieldStyle(edit, looksLikeExpression(edit->text()) ||
                                                edit->text().contains(QLatin1Char('$')));
            rowLayout->addWidget(edit, 1);
            rowLayout->addWidget(browse);
            form->addRow(label, row);
            continue;
        }

        // Arnold-style texture colour space (drives TX / OCIO → ACEScg).
        if (inputName == QLatin1String("colorspace")) {
            form->addRow(label, makeColorSpaceCombo(input.value, commit));
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
                    : (input.name == QLatin1String("dispersion_abbe"))
                          ? QStringLiteral("Abbe number Vd (Arnold dispersion_abbe). "
                                           "0 = off; typical glass 20–90. Lower = stronger rainbow. "
                                           "Range starts at 0 — negatives are invalid.")
                          : QString();
            const QString inputName = input.name;
            QWidget* row = makeFreeFloatSliderRow(value, lo, hi, [commit, inputType, isInt, inputName,
                                                                 hi](double v) {
                // Hard floor for Abbe — FreeFloat otherwise accepts any typed value.
                if (inputName == QLatin1String("dispersion_abbe") ||
                    inputName.endsWith(QLatin1String("_abbe"))) {
                    v = std::clamp(v, 0.0, hi > 0.0 ? hi : 200.0);
                }
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

    auto* scroll = new QScrollArea();
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setWidget(paramsBox);
    contentLayout_->addWidget(scroll, 1);
}

QWidget* ParameterPanel::createEditor(Parameter& parameter) {
    const QString name = parameter.name;
    Node* node = node_;
    auto notify = [this, node, name](const QVariant& value) {
        if (updating_ || !node) return;
        node->setParameterValue(name, value);
        emit parameterEdited(node, name);
        bool affectsVisibility = false;
        for (const Parameter& p : node->parameters()) {
            if (!p.visibleWhen.isEmpty() && p.visibleWhen.contains(name)) {
                affectsVisibility = true;
                break;
            }
        }
        if (affectsVisibility) QTimer::singleShot(0, this, [this] { refresh(); });
    };
    auto notifyText = [this, node, name](const QString& text) {
        if (updating_ || !node) return;
        Parameter* p = node->findParameter(name);
        if (!p) return;
        if (p->type == ParamType::Float || p->type == ParamType::Int || p->type == ParamType::String ||
            p->type == ParamType::FilePath) {
            if (looksLikeExpression(text) ||
                ((p->type == ParamType::String || p->type == ParamType::FilePath) &&
                 text.contains(QLatin1Char('$')))) {
                node->setParameterExpression(name, text);
            } else if (p->type == ParamType::Float) {
                bool ok = false;
                const double v = QLocale::c().toDouble(text, &ok);
                node->setParameterValue(name, ok ? v : 0.0);
            } else if (p->type == ParamType::Int) {
                bool ok = false;
                const double v = QLocale::c().toDouble(text, &ok);
                node->setParameterValue(name, ok ? int(std::lround(v)) : 0);
            } else {
                node->setParameterValue(name, text);
            }
        } else {
            node->setParameterValue(name, text);
        }
        emit parameterEdited(node, name);
        bool affectsVisibility = false;
        for (const Parameter& pp : node->parameters()) {
            if (!pp.visibleWhen.isEmpty() && pp.visibleWhen.contains(name)) {
                affectsVisibility = true;
                break;
            }
        }
        if (affectsVisibility) QTimer::singleShot(0, this, [this] { refresh(); });
    };

    switch (parameter.type) {
        case ParamType::Float: {
            const double lo = parameter.minValue;
            const double hi = parameter.maxValue;
            const double shown = parameter.hasExpression() ? parameter.evaluatedNumber()
                                                           : parameter.toDouble();
            QWidget* slider = makeFreeFloatSliderRow(shown, parameter.expression, lo, hi, notifyText);
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
            // Camera lens: common focal-length presets (mm).
            if (name == QLatin1String("focal") && node_ && node_->typeName() == QLatin1String("camera")) {
                auto* row = new QWidget();
                auto* layout = new QHBoxLayout(row);
                layout->setContentsMargins(0, 0, 0, 0);
                layout->setSpacing(6);
                layout->addWidget(slider, 1);

                static const double kFocalPresets[] = {10.0, 12.0, 14.0, 16.0, 24.0,  28.0,  35.0,  40.0,
                                                      50.0, 55.0, 65.0, 75.0, 85.0,  90.0, 100.0, 135.0};
                auto* combo = new QComboBox();
                combo->setToolTip("Focal length presets (mm)");
                combo->setMaximumWidth(86);
                combo->addItem(QStringLiteral("mm…"));
                const double current = parameter.toDouble();
                int matched = 0;
                for (double mm : kFocalPresets) {
                    combo->addItem(QStringLiteral("%1 mm").arg(int(std::lround(mm))), mm);
                    if (matched == 0 && std::fabs(current - mm) < 1e-4) matched = combo->count() - 1;
                }
                combo->setCurrentIndex(matched);
                connect(combo, QOverload<int>::of(&QComboBox::activated), this,
                        [this, notify, combo](int index) {
                            if (updating_ || index <= 0) return;
                            notify(combo->itemData(index).toDouble());
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
            const double shown = parameter.hasExpression() ? parameter.evaluatedNumber()
                                                           : double(parameter.toInt());
            return makeFreeFloatSliderRow(shown, parameter.expression, double(lo), double(hi), notifyText);
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
            if (parameter.name == QLatin1String("colorspace"))
                return makeColorSpaceCombo(parameter.hasExpression() ? parameter.expression
                                                                     : parameter.toString(),
                                           notifyText);
            auto* edit = new PathLineEdit(parameter.hasExpression() ? parameter.expression
                                                                    : parameter.toString());
            edit->setAcceptDrops(true);
            applyExpressionFieldStyle(edit, parameter.hasExpression() ||
                                                looksLikeExpression(edit->text()) ||
                                                edit->text().contains(QLatin1Char('$')));
            if (parameter.name == "pattern") {
                edit->setPlaceholderText("e.g. /geo/sphere1 — Ctrl+V or drop from Scene Graph");
                if (parameter.tooltip.isEmpty())
                    edit->setToolTip("Prim path or glob. Copy a prim with Ctrl+C in the Scene Graph, "
                                     "or drag its name into this field.");
            }
            edit->onCommitted = [notifyText, edit](const QString& text) {
                applyExpressionFieldStyle(edit, looksLikeExpression(text) || text.contains(QLatin1Char('$')));
                notifyText(text);
            };
            connect(edit, &QLineEdit::editingFinished, this, [notifyText, edit] {
                applyExpressionFieldStyle(edit, looksLikeExpression(edit->text()) ||
                                                    edit->text().contains(QLatin1Char('$')));
                notifyText(edit->text());
            });
            return edit;
        }
        case ParamType::FilePath: {
            auto* container = new QWidget();
            auto* layout = new QHBoxLayout(container);
            layout->setContentsMargins(0, 0, 0, 0);
            auto* edit = new QLineEdit(parameter.hasExpression() ? parameter.expression
                                                                 : parameter.toString());
            applyExpressionFieldStyle(edit, parameter.hasExpression() ||
                                                looksLikeExpression(edit->text()) ||
                                                edit->text().contains(QLatin1Char('$')));
            auto* browse = new QPushButton("...");
            browse->setFixedWidth(30);
            layout->addWidget(edit, 1);
            layout->addWidget(browse, 0);
            const QString filter = parameter.fileFilter;
            const bool saveMode = parameter.fileSaveMode;
            const bool dirMode = parameter.fileDirectoryMode;
            connect(edit, &QLineEdit::editingFinished, this, [notifyText, edit] {
                applyExpressionFieldStyle(edit, looksLikeExpression(edit->text()) ||
                                                    edit->text().contains(QLatin1Char('$')));
                notifyText(edit->text());
            });
            connect(browse, &QPushButton::clicked, this,
                    [this, edit, filter, notifyText, saveMode, dirMode] {
                        QString path;
                        if (dirMode) {
                            path = QFileDialog::getExistingDirectory(this, "Choose folder", edit->text());
                        } else if (saveMode) {
                            path = QFileDialog::getSaveFileName(this, "Save file", edit->text(), filter);
                        } else {
                            const auto picked = TextureFileDialog::getOpenTexture(
                                this, QStringLiteral("Choose file"), edit->text(),
                                filter.isEmpty()
                                    ? QStringLiteral(
                                          "Images (*.png *.jpg *.jpeg *.exr *.hdr *.tx *.tif *.tiff "
                                          "*.bmp *.webp);;All Files (*)")
                                    : filter);
                            path = picked.path;
                        }
                        if (path.isEmpty()) return;
                        edit->setText(path);
                        applyExpressionFieldStyle(edit, looksLikeExpression(path) ||
                                                            path.contains(QLatin1Char('$')));
                        notifyText(path);
                    });
            return container;
        }
        case ParamType::Button: {
            auto* button = new QPushButton(parameter.label);
            const QString paramName = parameter.name;
            connect(button, &QPushButton::clicked, this, [this, paramName] {
                if (!node_ || updating_ || materialXMode_) return;
                emit parameterAction(node_, paramName);
            });
            return button;
        }
        case ParamType::Menu: {
            auto* combo = new NoWheelComboBox();
            // Long lens catalogue names must not inflate the Parameters dock width.
            combo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
            combo->setMinimumContentsLength(10);
            combo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            combo->setMaximumWidth(320);
            for (const QString& item : parameter.menuItems) {
                // Show a shorter label; keep the full id as tooltip via item data.
                QString label = item;
                label.replace(QLatin1String("__"), QLatin1String(" / "));
                label.replace(QLatin1Char('_'), QLatin1Char(' '));
                combo->addItem(label, item);
                combo->setItemData(combo->count() - 1, item, Qt::ToolTipRole);
            }
            combo->setCurrentIndex(std::clamp(parameter.toInt(), 0, std::max(0, combo->count() - 1)));
            connect(combo, &QComboBox::currentIndexChanged, this, [this, notify, name](int index) {
                // Ignore teardown (-1) and empty combos — see ParameterPanel::rebuild.
                if (index < 0) return;
                notify(index);
                // Pixel Filter → fill recommended Filter Radius (artist can still override).
                if (name == QLatin1String("pixelfilter") && node_) {
                    const float radius = defaultFilterRadius(index);
                    node_->setParameterValue(QStringLiteral("filterradius"), double(radius));
                    emit parameterEdited(node_, QStringLiteral("filterradius"));
                    QTimer::singleShot(0, this, [this] { refresh(); });
                }
            });
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
