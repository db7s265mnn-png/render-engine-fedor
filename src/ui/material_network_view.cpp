#include "ui/material_network_view.h"

#include <QApplication>
#include <QClipboard>
#include <QContextMenuEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFocusEvent>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QHash>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QKeySequence>
#include <QLabel>
#include <QLineEdit>
#include <QLineF>
#include <QListWidget>
#include <QMap>
#include <QMenu>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QRegularExpression>
#include <QResizeEvent>
#include <QSet>
#include <QShowEvent>
#include <QSignalBlocker>
#include <QStyleOptionGraphicsItem>
#include <QTimer>
#include <QToolButton>
#include <QToolTip>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>
#include <limits>

#include "io/image_io.h"
#include "ui/texture_file_dialog.h"
#include "ui/graph_view_nav.h"
#include "io/materialx_graph.h"
#include "render/metal_spectra.h"
#include "nodes/node.h"
#include "nodes/node_graph.h"
#include "nodes/parameter.h"
#include "ui/material_wire_item.h"
#include "ui/node_icons.h"
#include "ui/theme.h"

namespace sol {
namespace {

constexpr qreal kLayoutScale = 80.0;
constexpr qreal kPortRadius = 4.8;
constexpr qreal kPortHitRadius = 11.0;
constexpr qreal kPortSnapRadius = 28.0;

const QVector<MaterialXNodeCatalogEntry>& catalogCache() {
    static const QVector<MaterialXNodeCatalogEntry> cache = listMaterialXNodeCatalog();
    return cache;
}

}  // namespace

// Searchable MaterialX node creation popup (same interaction as Scene Network Tab menu).
class MaterialXCreateMenu : public QWidget {
    Q_OBJECT
public:
    explicit MaterialXCreateMenu(QWidget* parent = nullptr) : QWidget(parent, Qt::Popup) {
        setObjectName("MaterialXCreateMenu");
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(6, 6, 6, 6);
        layout->setSpacing(4);
        search_ = new QLineEdit(this);
        search_->setPlaceholderText("Add MaterialX node...");
        layout->addWidget(search_);
        list_ = new QListWidget(this);
        list_->setUniformItemSizes(true);
        layout->addWidget(list_);
        setMinimumWidth(320);
        setMinimumHeight(360);
        connect(search_, &QLineEdit::textChanged, this, [this](const QString& text) { populate(text); });
        connect(search_, &QLineEdit::returnPressed, this, &MaterialXCreateMenu::accept);
        connect(list_, &QListWidget::itemActivated, this, &MaterialXCreateMenu::accept);
        connect(list_, &QListWidget::itemClicked, this, &MaterialXCreateMenu::accept);
    }

    void popupAt(QPoint globalPosition) {
        search_->clear();
        populate(QString());
        move(globalPosition);
        show();
        search_->setFocus();
    }

signals:
    void nodeChosen(const QString& category, const QString& type);

protected:
    void keyPressEvent(QKeyEvent* event) override {
        switch (event->key()) {
            case Qt::Key_Escape:
                hide();
                return;
            case Qt::Key_Down:
            case Qt::Key_Up: {
                const int direction = event->key() == Qt::Key_Down ? 1 : -1;
                int row = list_->currentRow();
                for (int i = 0; i < list_->count(); ++i) {
                    row += direction;
                    if (row < 0 || row >= list_->count()) break;
                    if (list_->item(row)->flags() & Qt::ItemIsSelectable) {
                        list_->setCurrentRow(row);
                        break;
                    }
                }
                return;
            }
            default:
                break;
        }
        QWidget::keyPressEvent(event);
    }

    void focusOutEvent(QFocusEvent* event) override {
        QWidget::focusOutEvent(event);
        hide();
    }

private:
    void populate(const QString& filter) {
        list_->clear();
        QString currentGroup;
        for (const MaterialXNodeCatalogEntry& entry : catalogCache()) {
            bool matchesVariant = false;
            if (!filter.isEmpty()) {
                for (const QString& variant : entry.typeVariants) {
                    if (variant.contains(filter, Qt::CaseInsensitive)) {
                        matchesVariant = true;
                        break;
                    }
                }
            }
            if (!filter.isEmpty() && !entry.label.contains(filter, Qt::CaseInsensitive) &&
                !entry.category.contains(filter, Qt::CaseInsensitive) &&
                !entry.type.contains(filter, Qt::CaseInsensitive) &&
                !entry.group.contains(filter, Qt::CaseInsensitive) && !matchesVariant)
                continue;
            if (currentGroup != entry.group) {
                currentGroup = entry.group;
                auto* header = new QListWidgetItem(entry.group.toUpper());
                header->setFlags(Qt::NoItemFlags);
                header->setForeground(theme::accent());
                list_->addItem(header);
            }
            auto* item = new QListWidgetItem("   " + entry.category);
            item->setData(Qt::UserRole, entry.category);
            item->setData(Qt::UserRole + 1, entry.type);
            const QString tip = entry.typeVariants.isEmpty()
                                    ? (entry.category + " / " + entry.type)
                                    : (entry.category + "  [" + entry.typeVariants.join(", ") + "]");
            item->setToolTip(tip);
            list_->addItem(item);
        }
        for (int i = 0; i < list_->count(); ++i) {
            if (list_->item(i)->flags() & Qt::ItemIsSelectable) {
                list_->setCurrentRow(i);
                break;
            }
        }
    }

    void accept() {
        QListWidgetItem* item = list_->currentItem();
        if (item && (item->flags() & Qt::ItemIsSelectable)) {
            const QString category = item->data(Qt::UserRole).toString();
            const QString type = item->data(Qt::UserRole + 1).toString();
            hide();
            if (!category.isEmpty()) emit nodeChosen(category, type);
            return;
        }
        hide();
    }

    QLineEdit* search_ = nullptr;
    QListWidget* list_ = nullptr;
};

namespace {

const MaterialXNodeCatalogEntry* findCatalogEntry(const QString& category) {
    const QVector<MaterialXNodeCatalogEntry>& catalog = catalogCache();
    for (const MaterialXNodeCatalogEntry& entry : catalog) {
        if (entry.category == category) return &entry;
    }
    return nullptr;
}

bool isKnownMaterialXCategory(const QString& category) {
    if (category.isEmpty() || category == "materialx" || category == "nodegraph" || category == "nodedef" ||
        category == "implementation" || category == "backdrop")
        return false;
    if (findCatalogEntry(category)) return true;
    // Keep previously hardcoded essentials even if libraries failed to load.
    return category == "standard_surface" || category == "ray_switch_shader" || category == "ray_switch" ||
           category == "surfacematerial" || category == "standard_volume" || category == "image" ||
           category == "constant" || category == "multiply" || category == "mix" || category == "normalmap" ||
           category == "bump" || category == "displacement" || category == "tiledimage" || category == "add" ||
           category == "texcoord" || category == "triplanarprojection";
}

QColor colorForCategory(const QString& category) {
    if (category == "image" || category == "tiledimage") return QColor(42, 132, 132);
    if (category == "standard_surface") return QColor(189, 116, 45);
    if (category == "standard_volume") return QColor(72, 140, 180);
    if (category == "ray_switch_shader" || category == "ray_switch") return QColor(72, 140, 160);
    if (category == "surfacematerial") return QColor(126, 82, 170);
    if (category == "normalmap" || category == "bump") return QColor(96, 101, 108);
    if (category == "displacement") return QColor(180, 110, 70);
    const MaterialXNodeCatalogEntry* entry = findCatalogEntry(category);
    if (entry) {
        if (entry->group.startsWith("PBR")) return QColor(189, 116, 45);
        if (entry->group == "Texture") return QColor(42, 132, 132);
        if (entry->group == "Geometric") return QColor(72, 120, 168);
        if (entry->group == "Procedural") return QColor(58, 140, 98);
        if (entry->group == "Color") return QColor(150, 90, 120);
        if (entry->group == "Lights") return QColor(180, 150, 60);
    }
    return QColor(96, 101, 108);
}

bool isConnectableInput(const QString& name, const QString& type) {
    Q_UNUSED(name);
    // Filenames and freeform strings are edited in Parameters, not via wires.
    return type != "filename" && type != "string";
}

// Arnold/USD-style loose MaterialX type compatibility for graph wires.
bool materialXTypesConnectable(const QString& sourceType, const QString& destType) {
    const QString s = sourceType.toLower();
    const QString d = destType.toLower();
    if (s.isEmpty() || d.isEmpty()) return true;
    if (s == d) return true;
    auto isFloatish = [](const QString& t) {
        return t == "float" || t == "integer" || t == "int";
    };
    auto isColorish = [](const QString& t) { return t.startsWith(QLatin1String("color")); };
    auto isVectorish = [](const QString& t) { return t.startsWith(QLatin1String("vector")); };
    // Booleans are not auto-wired from colour/vector patterns (was a soft-snap footgun).
    if (d == "boolean" || s == "boolean") return s == d;
    if (isFloatish(s) && (isFloatish(d) || isColorish(d) || isVectorish(d))) return true;
    if ((isColorish(s) || isVectorish(s)) && (isFloatish(d) || isColorish(d) || isVectorish(d))) return true;
    if (s == "surfaceshader" && d == "surfaceshader") return true;
    if (s == "volumeshader" && d == "volumeshader") return true;
    if (s == "material" && d == "material") return true;
    if (d == "displacementshader" &&
        (s == "displacementshader" || s == "float" || s == "vector3" || s == "color3"))
        return true;
    return false;
}

int connectionScore(const QString& sourceType, const QString& destName, const QString& destType, bool occupied) {
    if (!materialXTypesConnectable(sourceType, destType)) return -1;
    int score = (sourceType.toLower() == destType.toLower()) ? 100 : 20;
    const QString s = sourceType.toLower();
    const QString d = destType.toLower();
    const QString n = destName.toLower();
    // Prefer the Arnold "Diffuse → Color" port when wiring color/vector patterns.
    if ((s.startsWith("color") || s.startsWith("vector") || s == "float") && n == "base_color") score += 80;
    if (n == "base" && s != "float" && s != "integer" && s != "int") score -= 40;
    // Soft-snap: prefer exact types. Color/vector → float is allowed explicitly but
    // must not win soft-snap over colour ports (triplanar→roughness was a common miss).
    if ((s.startsWith("color") || s.startsWith("vector")) && d == "float") {
        score -= 55;
        if (n == "specular_roughness" || n == "metalness" || n == "specular" || n == "emission" ||
            n == "transmission" || n == "subsurface" || n == "specular_ior" || n == "subsurface_scale" ||
            n == "shadow_opacity")
            score -= 25;
    }
    if (s == "float" && (d.startsWith("color") || d.startsWith("vector"))) score -= 10;
    if ((s.startsWith("vector") || s.startsWith("color")) && n == "normal") score += 40;
    if (n == "displacement" || n == "displacementshader") {
        if (s == "float" || s == "vector3" || s == "color3" || s == "displacementshader") score += 80;
    }
    if ((n == "volume" || n == "volumeshader") && s == "volumeshader") score += 80;
    if (!occupied) score += 5;
    return score;
}

QString fallbackDefaultDocument() {
    return QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <standard_surface name=\"standard_surface1\" type=\"surfaceshader\" xpos=\"0\" ypos=\"0\">\n"
        "    <input name=\"base_color\" type=\"color3\" value=\"0.8, 0.8, 0.8\"/>\n"
        "    <input name=\"base\" type=\"float\" value=\"0.8\"/>\n"
        "    <input name=\"specular\" type=\"float\" value=\"0.5\"/>\n"
        "    <input name=\"specular_color\" type=\"color3\" value=\"1, 1, 1\"/>\n"
        "    <input name=\"specular_roughness\" type=\"float\" value=\"0.35\"/>\n"
        "    <input name=\"metalness\" type=\"float\" value=\"0\"/>\n"
        "    <input name=\"conductor_eta\" type=\"color3\" value=\"1.5, 1.5, 1.5\"/>\n"
        "    <input name=\"conductor_k\" type=\"color3\" value=\"0, 0, 0\"/>\n"
        "    <input name=\"transmission\" type=\"float\" value=\"0\"/>\n"
        "    <input name=\"transmission_color\" type=\"color3\" value=\"1, 1, 1\"/>\n"
        "    <input name=\"subsurface_scale\" type=\"float\" value=\"1\"/>\n"
        "    <input name=\"opacity\" type=\"color3\" value=\"1, 1, 1\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\" xpos=\"4\" ypos=\"0\">\n"
        "    <input name=\"surface\" type=\"surfaceshader\" nodename=\"standard_surface1\"/>\n"
        "    <input name=\"displacement\" type=\"displacementshader\"/>\n"
        "    <input name=\"volume\" type=\"volumeshader\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n");
}

QPointF defaultLayoutForCategory(const QString& category, int ordinal) {
    if (category == "surfacematerial") return QPointF(4.0, 0.0);
    if (category == "standard_surface") return QPointF(0.0, 0.0);
    if (category == "image") return QPointF(-4.0, qreal(ordinal) * 1.2);
    if (category == "normalmap" || category == "bump") return QPointF(-2.0, qreal(ordinal) * 1.2);
    if (category == "displacement") return QPointF(2.0, qreal(ordinal) * 1.2 + 1.5);
    return QPointF(-2.5, qreal(ordinal) * 1.2);
}

QPainterPath makeWirePath(QPointF from, QPointF to) {
    QPainterPath path(from);
    const qreal dx = std::max(qreal(48.0), std::abs(to.x() - from.x()) * 0.5);
    path.cubicTo(from + QPointF(dx, 0.0), to - QPointF(dx, 0.0), to);
    return path;
}

class MaterialNetworkNodeItem : public QGraphicsItem {
public:
    enum { Type = UserType + 71 };

    struct InputPort {
        QString name;
        QString type;
        int modelIndex = -1;
        bool connected = false;
    };

    MaterialNetworkNodeItem(MaterialNetworkGraphView* view, QString nodeName, QString category, QString typeName,
                            QVector<InputPort> inputs, QString subtitle)
        : view_(view),
          nodeName_(std::move(nodeName)),
          category_(std::move(category)),
          typeName_(std::move(typeName)),
          inputs_(std::move(inputs)),
          subtitle_(std::move(subtitle)) {
        setFlag(ItemIsSelectable, true);
        setFlag(ItemIsMovable, true);
        setFlag(ItemSendsGeometryChanges, true);
        setCacheMode(NoCache);
        // Arrow cursor like Scene Network — OpenHand looked like "drag only".
        setZValue(2.0);
    }

    int type() const override { return Type; }
    const QString& nodeName() const { return nodeName_; }
    const QString& typeName() const { return typeName_; }
    bool containsBody(QPointF scenePosition) const { return bodyRect().contains(mapFromScene(scenePosition)); }
    const QString& category() const { return category_; }
    QString inputPortName(int portIndex) const { return inputs_.value(portIndex).name; }
    QString inputPortType(int portIndex) const { return inputs_.value(portIndex).type; }
    int inputPortIndexByName(const QString& name) const {
        for (int i = 0; i < inputs_.size(); ++i) {
            if (inputs_[i].name == name) return i;
        }
        return -1;
    }

    QRectF boundingRect() const override { return bodyRect().adjusted(-16.0, -14.0, 16.0, 14.0); }

    QPainterPath shape() const override {
        QPainterPath path;
        path.addRoundedRect(bodyRect(), 6.0, 6.0);
        // Keep port hit pads small so they don't swallow body clicks (Scene Network
        // puts ports outside the tile; MaterialX ports sit on the edge).
        const qreal pad = kPortRadius + 3.0;
        for (int i = 0; i < inputs_.size(); ++i) path.addEllipse(inputPortLocal(i), pad, pad);
        path.addEllipse(outputPortLocal(), pad, pad);
        return path;
    }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) override {
        painter->setRenderHint(QPainter::Antialiasing, true);
        const QRectF body = bodyRect();
        painter->setPen(Qt::NoPen);
        painter->setBrush(QColor(0, 0, 0, 78));
        painter->drawRoundedRect(body.translated(2.0, 3.0), 6.0, 6.0);

        QLinearGradient gradient(body.topLeft(), body.bottomLeft());
        gradient.setColorAt(0.0, theme::panelLight().lighter(110));
        gradient.setColorAt(1.0, theme::panel().darker(112));
        painter->setBrush(gradient);
        QPen border(isSelected() ? theme::selection() : QColor(20, 21, 24), isSelected() ? 2.0 : 1.0);
        painter->setPen(border);
        painter->drawRoundedRect(body, 6.0, 6.0);

        painter->save();
        QPainterPath clip;
        clip.addRoundedRect(body, 6.0, 6.0);
        painter->setClipPath(clip);
        painter->setPen(Qt::NoPen);
        painter->setBrush(colorForCategory(category_));
        painter->drawRect(QRectF(body.left(), body.top(), body.width(), 20.0));

        // Official MaterialX mark in the header strip.
        const QPixmap mtlx = nodeIconPixmap(NodeIconKind::Material);
        if (!mtlx.isNull()) {
            const QRectF badge(body.right() - 16.0, body.top() + 4.0, 12.0, 12.0);
            painter->setRenderHint(QPainter::SmoothPixmapTransform, true);
            painter->drawPixmap(badge, mtlx, QRectF(mtlx.rect()));
        }
        painter->restore();

        QFont nameFont = painter->font();
        nameFont.setPointSizeF(8.2);
        nameFont.setBold(true);
        painter->setFont(nameFont);
        painter->setPen(theme::text());
        const QRectF nameRect(body.left() + 8.0, body.top() + 2.0, body.width() - 28.0, 16.0);
        painter->drawText(nameRect, Qt::AlignLeft | Qt::AlignVCenter,
                          QFontMetrics(nameFont).elidedText(nodeName_, Qt::ElideRight, int(nameRect.width())));

        QFont small = painter->font();
        small.setBold(false);
        small.setPointSizeF(7.0);
        painter->setFont(small);
        painter->setPen(theme::textDim());
        const QString typeLabel = subtitle_.isEmpty() ? category_ : subtitle_;
        const QRectF typeRect(body.left() + 8.0, body.top() + 21.0, body.width() - 16.0, 14.0);
        painter->drawText(typeRect, Qt::AlignLeft | Qt::AlignVCenter,
                          QFontMetrics(small).elidedText(typeLabel, Qt::ElideRight, int(typeRect.width())));

        for (int i = 0; i < inputs_.size(); ++i) {
            const QPointF port = inputPortLocal(i);
            const QRectF labelRect(port.x() + 9.0, port.y() - 8.0, body.width() - 20.0, 16.0);
            painter->drawText(labelRect, Qt::AlignLeft | Qt::AlignVCenter,
                              QFontMetrics(small).elidedText(inputs_[i].name, Qt::ElideRight,
                                                            int(labelRect.width())));
        }
        const QPointF out = outputPortLocal();
        painter->drawText(QRectF(out.x() - 34.0, out.y() - 8.0, 28.0, 16.0), Qt::AlignRight | Qt::AlignVCenter,
                          "out");

        paintPorts(painter);
    }

    QPointF inputPortScene(int portIndex) const { return mapToScene(inputPortLocal(portIndex)); }
    QPointF outputPortScene() const { return mapToScene(outputPortLocal()); }
    int inputCount() const { return inputs_.size(); }
    int inputModelIndex(int portIndex) const { return inputs_.value(portIndex).modelIndex; }

    int hitInputPort(QPointF scenePosition, qreal radius = kPortHitRadius) const {
        int best = -1;
        qreal bestDist = radius;
        for (int i = 0; i < inputs_.size(); ++i) {
            const qreal dist = QLineF(scenePosition, inputPortScene(i)).length();
            if (dist <= bestDist) {
                bestDist = dist;
                best = i;
            }
        }
        return best;
    }

    bool hitOutputPort(QPointF scenePosition, qreal radius = kPortHitRadius) const {
        return QLineF(scenePosition, outputPortScene()).length() <= radius;
    }

    QVariant itemChange(GraphicsItemChange change, const QVariant& value) override {
        if (change == ItemPositionHasChanged && view_) view_->updateWiresLive();
        return QGraphicsItem::itemChange(change, value);
    }

    qreal nearestInputDistance(QPointF scenePosition, int* portOut = nullptr) const {
        qreal best = 1e9;
        int bestPort = -1;
        for (int i = 0; i < inputs_.size(); ++i) {
            const qreal dist = QLineF(scenePosition, inputPortScene(i)).length();
            if (dist < best) {
                best = dist;
                bestPort = i;
            }
        }
        if (portOut) *portOut = bestPort;
        return best;
    }

private:
    QRectF bodyRect() const {
        const qreal width = category_ == "standard_surface" ? 172.0 : (category_ == "surfacematerial" ? 156.0 : 146.0);
        const qreal rowCount = qreal(std::max<qsizetype>(1, inputs_.size()));
        const qreal height = std::max(qreal(54.0), 40.0 + rowCount * 18.0);
        return QRectF(-width * 0.5, -height * 0.5, width, height);
    }

    QPointF inputPortLocal(int index) const {
        const QRectF body = bodyRect();
        const qreal y = body.top() + 43.0 + qreal(index) * 18.0;
        return QPointF(body.left() - 4.0, std::min(y, body.bottom() - 14.0));
    }

    QPointF outputPortLocal() const {
        const QRectF body = bodyRect();
        return QPointF(body.right() + 4.0, body.center().y());
    }

    void paintPorts(QPainter* painter) const {
        auto drawPort = [painter](QPointF local, bool connected, const QColor& color) {
            painter->setPen(Qt::NoPen);
            painter->setBrush(QColor(color.red(), color.green(), color.blue(), connected ? 90 : 48));
            painter->drawEllipse(local, kPortRadius + 3.0, kPortRadius + 3.0);
            painter->setPen(QPen(QColor(17, 18, 22), 1.0));
            painter->setBrush(connected ? color.lighter(118) : color);
            painter->drawEllipse(local, kPortRadius, kPortRadius);
        };

        for (int i = 0; i < inputs_.size(); ++i)
            drawPort(inputPortLocal(i), inputs_[i].connected, theme::colorForMaterialXType(inputs_[i].type));
        drawPort(outputPortLocal(), true, theme::colorForMaterialXType(typeName_));
    }

    MaterialNetworkGraphView* view_ = nullptr;
    QString nodeName_;
    QString category_;
    QString typeName_;
    QVector<InputPort> inputs_;
    QString subtitle_;
};

MaterialNetworkNodeItem* nodeItemAt(QGraphicsView* view, const QPoint& viewPosition) {
    const QList<QGraphicsItem*> items = view->items(viewPosition);
    for (QGraphicsItem* item : items) {
        if (auto* nodeItem = qgraphicsitem_cast<MaterialNetworkNodeItem*>(item)) return nodeItem;
    }
    return nullptr;
}

MaterialNetworkNodeItem* nodeItemByName(QGraphicsScene* scene, const QString& name) {
    for (QGraphicsItem* item : scene->items()) {
        if (auto* nodeItem = qgraphicsitem_cast<MaterialNetworkNodeItem*>(item)) {
            if (nodeItem->nodeName() == name) return nodeItem;
        }
    }
    return nullptr;
}

MaterialNetworkNodeItem* outputPortAt(QGraphicsScene* scene, QPointF scenePosition, qreal radius = kPortSnapRadius) {
    MaterialNetworkNodeItem* best = nullptr;
    qreal bestDist = radius;
    for (QGraphicsItem* item : scene->items()) {
        auto* nodeItem = qgraphicsitem_cast<MaterialNetworkNodeItem*>(item);
        if (!nodeItem) continue;
        const qreal dist = QLineF(scenePosition, nodeItem->outputPortScene()).length();
        if (dist <= bestDist) {
            bestDist = dist;
            best = nodeItem;
        }
    }
    return best;
}

struct InputHit {
    MaterialNetworkNodeItem* item = nullptr;
    int inputIndex = -1;
};

InputHit inputPortAt(QGraphicsScene* scene, QPointF scenePosition, qreal radius = kPortSnapRadius) {
    InputHit best;
    qreal bestDist = radius;
    for (QGraphicsItem* item : scene->items()) {
        auto* nodeItem = qgraphicsitem_cast<MaterialNetworkNodeItem*>(item);
        if (!nodeItem) continue;
        int port = -1;
        const qreal dist = nodeItem->nearestInputDistance(scenePosition, &port);
        if (port >= 0 && dist <= bestDist) {
            bestDist = dist;
            best = {nodeItem, port};
        }
    }
    return best;
}

MaterialWireItem* wireItemAt(QGraphicsView* view, const QPoint& viewPosition) {
    for (QGraphicsItem* item : view->items(viewPosition)) {
        if (auto* wire = qgraphicsitem_cast<MaterialWireItem*>(item)) return wire;
    }
    // Fat hit-test via shape even when the thin stroke misses the exact pixel.
    const QPointF scenePosition = view->mapToScene(viewPosition);
    MaterialWireItem* best = nullptr;
    qreal bestDist = 10.0;
    for (QGraphicsItem* item : view->scene()->items()) {
        auto* wire = qgraphicsitem_cast<MaterialWireItem*>(item);
        if (!wire) continue;
        if (!wire->shape().contains(wire->mapFromScene(scenePosition))) continue;
        const qreal dist = QLineF(scenePosition, wire->path().pointAtPercent(0.5)).length();
        if (dist < bestDist || !best) {
            best = wire;
            bestDist = dist;
        }
    }
    return best;
}

void addWire(QGraphicsScene* scene, QPointF from, QPointF to, const QString& sourceName, const QString& targetName,
             const QString& inputName, const QColor& color = QColor()) {
    auto* wire = new MaterialWireItem(sourceName, targetName, inputName, color);
    wire->setWirePath(makeWirePath(from, to));
    scene->addItem(wire);
}

}  // namespace

MaterialNetworkGraphView::MaterialNetworkGraphView(QWidget* parent) : QGraphicsView(parent) {
    graphScene_ = new QGraphicsScene(this);
    // Large scene rect (same idea as Scene Network) so pan works freely on both axes.
    graphScene_->setSceneRect(-8000, -8000, 16000, 16000);
    setScene(graphScene_);
    setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);
    // Full updates avoid antialiased trails while dragging MaterialX nodes.
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    setTransformationAnchor(QGraphicsView::NoAnchor);
    setResizeAnchor(QGraphicsView::NoAnchor);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setDragMode(QGraphicsView::RubberBandDrag);
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
    setMinimumHeight(180);
    lastMousePoint_ = viewport()->rect().center();
    selectionConnection_ = connect(graphScene_, &QGraphicsScene::selectionChanged, this,
                                   &MaterialNetworkGraphView::emitSelectionChanged);

    createMenu_ = new MaterialXCreateMenu(this);
    connect(createMenu_, &MaterialXCreateMenu::nodeChosen, this,
            &MaterialNetworkGraphView::onCreateMenuChosen);
    rebuild();
}

void MaterialNetworkGraphView::setMaterialNode(Node* node) {
    if (node && node->typeName() != "material") node = nullptr;
    if (materialNode_ == node) return;

    if (materialChangedConnection_) disconnect(materialChangedConnection_);
    materialChangedConnection_ = {};
    materialNode_ = node;

    if (materialNode_) {
        ensureMtlxParameter();
        QString xml = materialNode_->stringValue("mtlx");
        if (xml.trimmed().isEmpty()) {
            xml = defaultDocument();
            writeXmlToMaterial(xml, false);
        } else {
            const QString normalized = normalizeMaterialXDocument(xml);
            if (!normalized.trimmed().isEmpty() && normalized != xml) {
                xml = normalized;
                writeXmlToMaterial(xml, false);
            }
        }

        materialChangedConnection_ =
            connect(materialNode_, &Node::parameterChanged, this, [this](Node* node, const QString& parameterName) {
                if (suppressMaterialSignal_) return;
                if (node == materialNode_ && parameterName == "mtlx") rebuild();
            });

        if (!materialXAvailable()) emit statusMessage("MaterialX libraries unavailable; editing stored XML directly");
    }

    rebuild();
}

void MaterialNetworkGraphView::ensureMtlxParameter() {
    if (!materialNode_ || materialNode_->findParameter("mtlx")) return;
    materialNode_->addParameter(
        Parameter::makeString("mtlx", "MaterialX XML", QString()).withGroup("MaterialX").withTooltip(
            "Stored MaterialX XML edited by the Material Network view"));
}

QString MaterialNetworkGraphView::defaultDocument() const {
    const QString helperXml = createDefaultMaterialXDocument();
    return helperXml.trimmed().isEmpty() ? fallbackDefaultDocument() : helperXml;
}

void MaterialNetworkGraphView::writeXmlToMaterial(const QString& xml, bool emitEdited) {
    if (!materialNode_) return;
    ensureMtlxParameter();
    suppressMaterialSignal_ = true;
    // notify=false when not emitting — avoids graphChanged → cook for layout/repair writes.
    materialNode_->setParameterValue("mtlx", xml, emitEdited);
    suppressMaterialSignal_ = false;
    if (emitEdited) emit materialEdited(materialNode_);
}

MaterialNetworkGraphView::MtlxNode* MaterialNetworkGraphView::findModelNode(const QString& name) {
    for (MtlxNode& node : graphNodes_) {
        if (node.name == name) return &node;
    }
    return nullptr;
}

const MaterialNetworkGraphView::MtlxNode* MaterialNetworkGraphView::findModelNode(const QString& name) const {
    for (const MtlxNode& node : graphNodes_) {
        if (node.name == name) return &node;
    }
    return nullptr;
}

QString MaterialNetworkGraphView::uniqueNodeName(const QString& baseName) const {
    auto exists = [this](const QString& name) {
        return std::any_of(graphNodes_.begin(), graphNodes_.end(),
                           [&name](const MtlxNode& node) { return node.name == name; });
    };
    if (!exists(baseName)) return baseName;
    for (int i = 1; i < 10000; ++i) {
        const QString candidate = baseName + QString::number(i);
        if (!exists(candidate)) return candidate;
    }
    return baseName + QString::number(graphNodes_.size() + 1);
}

void MaterialNetworkGraphView::ensureInput(QVector<MtlxInput>& inputs, const QString& name, const QString& type,
                                      const QString& value) {
    for (MtlxInput& input : inputs) {
        if (input.name == name) {
            if (input.type.isEmpty()) input.type = type;
            return;
        }
    }
    inputs.push_back({name, type, value, {}});
}

// Stable UI / wire port order — independent of MaterialX nodedef or XML authorship order.
QStringList MaterialNetworkGraphView::canonicalInputOrder(const QString& category) {
    if (category == "standard_surface") {
        return {QStringLiteral("base_color"),
                QStringLiteral("base"),
                QStringLiteral("specular"),
                QStringLiteral("specular_color"),
                QStringLiteral("specular_roughness"),
                QStringLiteral("specular_IOR"),
                QStringLiteral("metalness"),
                QStringLiteral("conductor_eta"),
                QStringLiteral("conductor_k"),
                QStringLiteral("transmission"),
                QStringLiteral("transmission_color"),
                QStringLiteral("shadow_opacity"),
                QStringLiteral("contribute_caustics"),
                QStringLiteral("dispersion_abbe"),
                QStringLiteral("thin_film_thickness"),
                QStringLiteral("thin_film_IOR"),
                QStringLiteral("internal_reflections"),
                QStringLiteral("emission"),
                QStringLiteral("emission_color"),
                QStringLiteral("normal"),
                QStringLiteral("subsurface"),
                QStringLiteral("subsurface_color"),
                QStringLiteral("subsurface_radius"),
                QStringLiteral("subsurface_scale"),
                QStringLiteral("opacity")};
    }
    if (category == "ray_switch_shader" || category == "ray_switch") {
        return {QStringLiteral("camera"),
                QStringLiteral("shadow"),
                QStringLiteral("diffuse_reflection"),
                QStringLiteral("specular_reflection"),
                QStringLiteral("diffuse_transmission"),
                QStringLiteral("specular_transmission"),
                QStringLiteral("sss"),
                QStringLiteral("caustics")};
    }
    if (category == "surfacematerial") {
        return {QStringLiteral("surface"), QStringLiteral("displacement"), QStringLiteral("volume")};
    }
    if (category == "standard_volume") {
        return {QStringLiteral("density"),
                QStringLiteral("anisotropy"),
                QStringLiteral("absorption"),
                QStringLiteral("scattering"),
                QStringLiteral("emission"),
                QStringLiteral("emission_color")};
    }
    if (category == "triplanarprojection") {
        // Arnold Triplanar: Input → Input Per Axis → axis files → Transform → Blend.
        return {QStringLiteral("file"),
                QStringLiteral("input_per_axis"),
                QStringLiteral("filex"),
                QStringLiteral("filey"),
                QStringLiteral("filez"),
                QStringLiteral("scale"),
                QStringLiteral("rotate"),
                QStringLiteral("offset"),
                QStringLiteral("blend"),
                QStringLiteral("default")};
    }
    return {};
}

void MaterialNetworkGraphView::normalizeInputOrder(QVector<MtlxInput>& inputs, const QString& category) {
    const QStringList order = canonicalInputOrder(category);
    if (order.isEmpty() || inputs.isEmpty()) return;
    QHash<QString, MtlxInput> byName;
    byName.reserve(inputs.size());
    for (const MtlxInput& input : inputs) {
        if (!input.name.isEmpty()) byName.insert(input.name, input);
    }
    QVector<MtlxInput> out;
    out.reserve(order.size());
    for (const QString& name : order) {
        const auto it = byName.constFind(name);
        if (it != byName.constEnd()) out.push_back(*it);
    }
    inputs = out;
}

QString MaterialNetworkGraphView::defaultTypeForCategory(const QString& category) {
    if (const MaterialXNodeCatalogEntry* entry = findCatalogEntry(category)) return entry->type;
    if (category == "standard_surface") return "surfaceshader";
    if (category == "standard_volume") return "volumeshader";
    if (category == "surfacematerial") return "material";
    if (category == "normalmap") return "vector3";
    if (category == "bump") return "vector3";
    if (category == "displacement") return "float";
    if (category == "texcoord") return "vector2";
    return "color3";
}

QVector<MaterialNetworkGraphView::MtlxInput> MaterialNetworkGraphView::defaultInputsForCategory(const QString& category,
                                                                                      const QString& type) {
    QVector<MtlxInput> inputs;
    if (category == "standard_surface") {
        // Always emit Arnold Diffuse Color first — never follow nodedef/XML order.
        static const QVector<MtlxInput> kDefaults = {
            {"base_color", "color3", "0.8, 0.8, 0.8", {}},
            {"base", "float", "0.8", {}},
            {"specular", "float", "0.5", {}},
            {"specular_color", "color3", "1, 1, 1", {}},
            {"specular_roughness", "float", "0.35", {}},
            {"specular_IOR", "float", "1.5", {}},
            {"metalness", "float", "0", {}},
            {"conductor_eta", "color3", "1.5, 1.5, 1.5", {}},
            {"conductor_k", "color3", "0, 0, 0", {}},
            {"transmission", "float", "0", {}},
            {"transmission_color", "color3", "1, 1, 1", {}},
            {"shadow_opacity", "float", "1", {}},
            {"contribute_caustics", "boolean", "true", {}},
            {"dispersion_abbe", "float", "0", {}},
            {"thin_film_thickness", "float", "0", {}},
            {"thin_film_IOR", "float", "1.4", {}},
            {"internal_reflections", "boolean", "true", {}},
            {"emission", "float", "0", {}},
            {"emission_color", "color3", "1, 1, 1", {}},
            {"normal", "vector3", {}, {}},
            {"subsurface", "float", "0", {}},
            {"subsurface_color", "color3", "1, 0.75, 0.55", {}},
            {"subsurface_radius", "color3", "1, 0.35, 0.2", {}},
            {"subsurface_scale", "float", "1", {}},
            {"opacity", "color3", "1, 1, 1", {}},
        };
        if (const MaterialXNodeCatalogEntry* entry = findCatalogEntry(category)) {
            const QString signature = type.isEmpty() ? entry->type : type;
            QHash<QString, MaterialXNodeInputDef> byName;
            for (const MaterialXNodeInputDef& def : entry->inputsFor(signature)) byName.insert(def.name, def);
            for (const MtlxInput& fallback : kDefaults) {
                const auto it = byName.constFind(fallback.name);
                if (it != byName.constEnd())
                    inputs.push_back({it->name, it->type, it->value.isEmpty() ? fallback.value : it->value, {}});
                else
                    inputs.push_back(fallback);
            }
        } else {
            inputs = kDefaults;
        }
        return inputs;
    }

    if (category == "ray_switch_shader") {
        return {{"camera", "surfaceshader", {}, {}},
                {"shadow", "surfaceshader", {}, {}},
                {"diffuse_reflection", "surfaceshader", {}, {}},
                {"specular_reflection", "surfaceshader", {}, {}},
                {"diffuse_transmission", "surfaceshader", {}, {}},
                {"specular_transmission", "surfaceshader", {}, {}},
                {"sss", "surfaceshader", {}, {}},
                {"caustics", "surfaceshader", {}, {}}};
    }
    if (category == "ray_switch") {
        return {{"camera", "color3", "0.8, 0.8, 0.8", {}},
                {"shadow", "color3", "0.8, 0.8, 0.8", {}},
                {"diffuse_reflection", "color3", "0.8, 0.8, 0.8", {}},
                {"specular_reflection", "color3", "0.8, 0.8, 0.8", {}},
                {"diffuse_transmission", "color3", "0.8, 0.8, 0.8", {}},
                {"specular_transmission", "color3", "0.8, 0.8, 0.8", {}},
                {"sss", "color3", "0.8, 0.8, 0.8", {}},
                {"caustics", "color3", "0.8, 0.8, 0.8", {}}};
    }

    if (category == "triplanarprojection") {
        // Arnold Triplanar: shared Input by default; per-axis unlocked by checkbox.
        QVector<MtlxInput> out = {
            {"file", "filename", {}, {}},
            {"input_per_axis", "boolean", "false", {}},
            {"filex", "filename", {}, {}},
            {"filey", "filename", {}, {}},
            {"filez", "filename", {}, {}},
            {"scale", "vector3", "1, 1, 1", {}},
            {"rotate", "float", "0", {}},
            {"offset", "vector3", "0, 0, 0", {}},
            {"blend", "float", "0.1", {}},
            {"default", "color3", "0.2, 0.5, 0.8", {}},
        };
        if (!type.isEmpty() && type != "color3") {
            for (MtlxInput& input : out) {
                if (input.name == "default") {
                    input.type = type;
                    input.value = type.startsWith("color") ? QString("0.2, 0.5, 0.8") : QString("0, 0, 0");
                }
            }
        }
        if (const MaterialXNodeCatalogEntry* entry = findCatalogEntry(category)) {
            const QString signature = type.isEmpty() ? entry->type : type;
            QHash<QString, MaterialXNodeInputDef> byName;
            for (const MaterialXNodeInputDef& def : entry->inputsFor(signature)) byName.insert(def.name, def);
            for (MtlxInput& input : out) {
                const auto it = byName.constFind(input.name);
                if (it == byName.constEnd()) continue;
                input.type = it->type.isEmpty() ? input.type : it->type;
                if (input.value.isEmpty() && !it->value.isEmpty()) input.value = it->value;
            }
        }
        return out;
    }

    if (category == "surfacematerial") {
        QVector<MtlxInput> out = {
            {"surface", "surfaceshader", {}, {}},
            {"displacement", "displacementshader", {}, {}},
            {"volume", "volumeshader", {}, {}},
        };
        if (const MaterialXNodeCatalogEntry* entry = findCatalogEntry(category)) {
            const QString signature = type.isEmpty() ? entry->type : type;
            QHash<QString, MaterialXNodeInputDef> byName;
            for (const MaterialXNodeInputDef& def : entry->inputsFor(signature)) byName.insert(def.name, def);
            for (MtlxInput& input : out) {
                const auto it = byName.constFind(input.name);
                if (it == byName.constEnd()) continue;
                input.type = it->type.isEmpty() ? input.type : it->type;
            }
        }
        return out;
    }

    if (category == "standard_volume") {
        QVector<MtlxInput> out = {
            {"density", "float", "1", {}},
            {"anisotropy", "float", "0", {}},
            {"absorption", "color3", "0, 0, 0", {}},
            {"scattering", "color3", "1, 1, 1", {}},
            {"emission", "float", "0", {}},
            {"emission_color", "color3", "1, 1, 1", {}},
        };
        if (const MaterialXNodeCatalogEntry* entry = findCatalogEntry(category)) {
            const QString signature = type.isEmpty() ? entry->type : type;
            QHash<QString, MaterialXNodeInputDef> byName;
            for (const MaterialXNodeInputDef& def : entry->inputsFor(signature)) byName.insert(def.name, def);
            for (MtlxInput& input : out) {
                const auto it = byName.constFind(input.name);
                if (it == byName.constEnd()) continue;
                input.type = it->type.isEmpty() ? input.type : it->type;
                if (input.value.isEmpty() && !it->value.isEmpty()) input.value = it->value;
            }
        }
        return out;
    }

    if (category == "displacement") {
        const QString t = (type == "vector3") ? QString("vector3") : QString("float");
        QVector<MtlxInput> out;
        if (t == "vector3")
            out.push_back({"displacement", "vector3", "0, 0, 0", {}});
        else
            out.push_back({"displacement", "float", "0", {}});
        out.push_back({"scale", "float", "1", {}});
        out.push_back({"autobump", "boolean", "true", {}});
        out.push_back({"zero_value", "float", "0.5", {}});
        if (const MaterialXNodeCatalogEntry* entry = findCatalogEntry(category)) {
            const QString signature = type.isEmpty() ? entry->type : type;
            QHash<QString, MaterialXNodeInputDef> byName;
            for (const MaterialXNodeInputDef& def : entry->inputsFor(signature)) byName.insert(def.name, def);
            for (MtlxInput& input : out) {
                const auto it = byName.constFind(input.name);
                if (it == byName.constEnd()) continue;
                input.type = it->type.isEmpty() ? input.type : it->type;
                if (input.value.isEmpty() && !it->value.isEmpty()) input.value = it->value;
            }
        }
        return out;
    }

    if (const MaterialXNodeCatalogEntry* entry = findCatalogEntry(category)) {
        const QString signature = type.isEmpty() ? entry->type : type;
        for (const MaterialXNodeInputDef& def : entry->inputsFor(signature)) {
            inputs.push_back({def.name, def.type, def.value, {}});
        }
        if (!inputs.isEmpty()) return inputs;
    }

    if (category == "image" || category == "tiledimage") {
        inputs.push_back({"file", "filename", {}, {}});
    } else if (category == "constant") {
        inputs.push_back({"value", type.isEmpty() ? QString("color3") : type, "1, 1, 1", {}});
    } else if (category == "multiply" || category == "add") {
        const QString t = type.isEmpty() ? QString("color3") : type;
        inputs.push_back({"in1", t, t.startsWith("color") || t.startsWith("vector") ? "1, 1, 1" : "1", {}});
        inputs.push_back({"in2", t, t.startsWith("color") || t.startsWith("vector") ? "1, 1, 1" : "1", {}});
    } else if (category == "mix") {
        const QString t = type.isEmpty() ? QString("color3") : type;
        inputs.push_back({"bg", t, "0, 0, 0", {}});
        inputs.push_back({"fg", t, "1, 1, 1", {}});
        inputs.push_back({"mix", "float", "0.5", {}});
    } else if (category == "normalmap") {
        inputs.push_back({"in", "vector3", {}, {}});
        inputs.push_back({"scale", "float", "1", {}});
    } else if (category == "bump") {
        inputs.push_back({"height", "float", "0", {}});
        inputs.push_back({"scale", "float", "1", {}});
    } else if (category == "displacement") {
        const QString t = type.isEmpty() ? QString("float") : type;
        if (t == "vector3")
            inputs.push_back({"displacement", "vector3", "0, 0, 0", {}});
        else
            inputs.push_back({"displacement", "float", "0", {}});
        inputs.push_back({"scale", "float", "1", {}});
        inputs.push_back({"autobump", "boolean", "true", {}});
        inputs.push_back({"zero_value", "float", "0.5", {}});
    } else if (category == "surfacematerial") {
        inputs.push_back({"surface", "surfaceshader", {}, {}});
        inputs.push_back({"displacement", "displacementshader", {}, {}});
        inputs.push_back({"volume", "volumeshader", {}, {}});
    } else if (category == "standard_volume") {
        inputs.push_back({"density", "float", "1", {}});
        inputs.push_back({"anisotropy", "float", "0", {}});
        inputs.push_back({"absorption", "color3", "0, 0, 0", {}});
        inputs.push_back({"scattering", "color3", "1, 1, 1", {}});
        inputs.push_back({"emission", "float", "0", {}});
        inputs.push_back({"emission_color", "color3", "1, 1, 1", {}});
    }
    return inputs;
}

void MaterialNetworkGraphView::rebuildFromXml(const QString& xml, bool rewriteRepaired) {
    graphNodes_.clear();
    udimSet_.clear();
    bool repaired = false;
    int ordinal = 0;

    // Parse through MaterialX (not Qt XML): MaterialX keeps <UDIM> unescaped in
    // attribute values, which QXmlStreamReader rejects as malformed markup.
    QVector<MaterialXGraphNode> parsed;
    QString parseError;
    const bool parsedOk = parseMaterialXGraph(xml, parsed, &parseError, &udimSet_);
    if (!parsedOk) {
        if (!parseError.isEmpty()) emit statusMessage("MaterialX parse: " + parseError);
        repaired = true;
    } else {
        for (const MaterialXGraphNode& parsedNode : parsed) {
            if (!isKnownMaterialXCategory(parsedNode.category)) continue;
            MtlxNode node;
            node.category = parsedNode.category;
            node.name = parsedNode.name;
            if (node.name.isEmpty()) {
                node.name = uniqueNodeName(parsedNode.category);
                repaired = true;
            }
            node.type = parsedNode.type.isEmpty() ? defaultTypeForCategory(node.category) : parsedNode.type;
            if (std::isnan(parsedNode.xpos) || std::isnan(parsedNode.ypos))
                node.layout = defaultLayoutForCategory(node.category, ordinal);
            else
                node.layout = QPointF(parsedNode.xpos, parsedNode.ypos);
            for (const MaterialXGraphInput& parsedInput : parsedNode.inputs) {
                if (parsedInput.name.isEmpty()) continue;
                MtlxInput input{parsedInput.name, parsedInput.type, parsedInput.value, parsedInput.nodename};
                // MaterialX authoring: concrete tile name.1001.exr → unresolved name.<UDIM>.exr
                if (input.nodename.isEmpty() && input.type == QLatin1String("filename") &&
                    !input.value.isEmpty() && !pathHasUdimToken(input.value)) {
                    QString pattern;
                    std::vector<int> tiles;
                    if (resolveUdimPattern(input.value, QString(), pattern, tiles)) {
                        input.value = pattern;
                        repaired = true;
                        for (int udim : tiles) {
                            if (!udimSet_.contains(udim)) udimSet_.push_back(udim);
                        }
                    }
                }
                node.inputs.push_back(input);
            }
            // Migrate MaterialX long port names → Solstice short UI names.
            if (node.category == QLatin1String("surfacematerial")) {
                auto renamePort = [&](const QString& from, const QString& to) {
                    for (MtlxInput& input : node.inputs) {
                        if (input.name != from) continue;
                        input.name = to;
                        repaired = true;
                    }
                };
                renamePort(QStringLiteral("surfaceshader"), QStringLiteral("surface"));
                renamePort(QStringLiteral("displacementshader"), QStringLiteral("displacement"));
                renamePort(QStringLiteral("volumeshader"), QStringLiteral("volume"));
            }
            for (const MtlxInput& input : defaultInputsForCategory(node.category, node.type))
                ensureInput(node.inputs, input.name, input.type, input.value);
            normalizeInputOrder(node.inputs, node.category);
            graphNodes_.push_back(node);
            ++ordinal;
        }
        const QVector<int> authoredUdims = udimSet_;
        refreshUdimSetFromFilenames();
        if (udimSet_ != authoredUdims) repaired = true;
    }

    if (graphNodes_.isEmpty()) {
        MtlxNode standard;
        standard.name = "standard_surface1";
        standard.category = "standard_surface";
        standard.type = "surfaceshader";
        standard.layout = QPointF(0.0, 0.0);
        standard.inputs = defaultInputsForCategory(standard.category);
        graphNodes_.push_back(standard);

        MtlxNode surface;
        surface.name = "surface";
        surface.category = "surfacematerial";
        surface.type = "material";
        surface.layout = QPointF(4.0, 0.0);
        surface.inputs = defaultInputsForCategory(surface.category);
        surface.inputs[0].nodename = standard.name;
        graphNodes_.push_back(surface);
        repaired = true;
    }

    MtlxNode* standard = nullptr;
    for (MtlxNode& node : graphNodes_) {
        if (node.category == "standard_surface") {
            standard = &node;
            break;
        }
    }
    if (!standard) {
        MtlxNode node;
        node.name = uniqueNodeName("standard_surface1");
        node.category = "standard_surface";
        node.type = "surfaceshader";
        node.layout = QPointF(0.0, 0.0);
        node.inputs = defaultInputsForCategory(node.category);
        graphNodes_.push_back(node);
        standard = &graphNodes_.back();
        repaired = true;
    }

    MtlxNode* surface = findModelNode("surface");
    if (!surface || surface->category != "surfacematerial") {
        MtlxNode node;
        node.name = surface ? uniqueNodeName("surface") : QString("surface");
        node.category = "surfacematerial";
        node.type = "material";
        node.layout = QPointF(4.0, 0.0);
        node.inputs = defaultInputsForCategory(node.category);
        graphNodes_.push_back(node);
        surface = &graphNodes_.back();
        repaired = true;
    }
    ensureInput(surface->inputs, "surface", "surfaceshader");
    ensureInput(surface->inputs, "displacement", "displacementshader");
    ensureInput(surface->inputs, "volume", "volumeshader");
    MtlxInput* surfaceShaderInput = nullptr;
    for (MtlxInput& input : surface->inputs) {
        if (input.name == "surface" || input.name == "surfaceshader") {
            surfaceShaderInput = &input;
            break;
        }
    }
    if (surfaceShaderInput && surfaceShaderInput->nodename.isEmpty()) {
        surfaceShaderInput->nodename = standard->name;
        surfaceShaderInput->value.clear();
        repaired = true;
    }

    if (rewriteRepaired && repaired) writeModel(false);
}

void MaterialNetworkGraphView::rebuild() {
    if (preservedSelection_.isEmpty()) preservedSelection_ = selectedNodeName();
    const bool blocked = graphScene_->blockSignals(true);
    graphScene_->clear();
    previewWire_ = nullptr;
    graphNodes_.clear();

    if (!materialNode_) {
        graphScene_->setSceneRect(-8000, -8000, 16000, 16000);
        graphScene_->blockSignals(blocked);
        viewport()->update();
        emitSelectionChanged();
        return;
    }

    ensureMtlxParameter();
    QString xml = materialNode_->stringValue("mtlx");
    if (xml.trimmed().isEmpty()) {
        xml = defaultDocument();
        writeXmlToMaterial(xml, false);
    }
    rebuildFromXml(xml, true);

    for (const MtlxNode& node : graphNodes_) {
        QVector<MaterialNetworkNodeItem::InputPort> ports;
        for (int i = 0; i < node.inputs.size(); ++i) {
            const MtlxInput& input = node.inputs[i];
            if (!isConnectableInput(input.name, input.type)) continue;
            ports.push_back({input.name, input.type, i, !input.nodename.isEmpty()});
        }

        QString subtitle = node.category;
        if (node.category == "image" || node.category == "tiledimage") {
            for (const MtlxInput& input : node.inputs) {
                if (input.name == "file") {
                    subtitle = input.value.trimmed().isEmpty() ? QString("choose texture")
                                                               : QFileInfo(input.value).fileName();
                    break;
                }
            }
        } else if (!node.type.isEmpty()) {
            subtitle = node.category + " / " + node.type;
        }

        auto* item = new MaterialNetworkNodeItem(this, node.name, node.category, node.type, ports, subtitle);
        item->setPos(node.layout * kLayoutScale);
        graphScene_->addItem(item);
    }

    for (const MtlxNode& target : graphNodes_) {
        MaterialNetworkNodeItem* targetItem = nodeItemByName(graphScene_, target.name);
        if (!targetItem) continue;
        int visiblePort = 0;
        for (int i = 0; i < target.inputs.size(); ++i) {
            const MtlxInput& input = target.inputs[i];
            if (!isConnectableInput(input.name, input.type)) continue;
            if (!input.nodename.isEmpty()) {
                MaterialNetworkNodeItem* sourceItem = nodeItemByName(graphScene_, input.nodename);
                if (sourceItem)
                    addWire(graphScene_, sourceItem->outputPortScene(), targetItem->inputPortScene(visiblePort),
                            sourceItem->nodeName(), target.name, input.name,
                            theme::colorForMaterialXType(sourceItem->typeName()));
            }
            ++visiblePort;
        }
    }

    if (!preservedSelection_.isEmpty()) {
        if (MaterialNetworkNodeItem* item = nodeItemByName(graphScene_, preservedSelection_)) {
            item->setSelected(true);
        }
    }
    preservedSelection_.clear();

    // Keep a large scene rect so sticky pan is not clamped to content bounds.
    graphScene_->setSceneRect(-8000, -8000, 16000, 16000);
    graphScene_->blockSignals(blocked);
    scheduleFrameGraph();
    emitSelectionChanged();
}

QString MaterialNetworkGraphView::serializeGraph() const {
    // Serialize with MaterialX so filename tokens like <UDIM> stay valid MTLX
    // (angle brackets intentionally unescaped — Qt XML must not round-trip this).
    QVector<MaterialXGraphNode> nodes;
    nodes.reserve(graphNodes_.size());
    for (const MtlxNode& node : graphNodes_) {
        MaterialXGraphNode out;
        out.name = node.name;
        out.category = node.category;
        out.type = node.type.isEmpty() ? defaultTypeForCategory(node.category) : node.type;
        out.xpos = node.layout.x();
        out.ypos = node.layout.y();
        for (const MtlxInput& input : node.inputs) {
            if (input.name.isEmpty()) continue;
            out.inputs.push_back({input.name, input.type, input.value, input.nodename});
        }
        nodes.push_back(out);
    }
    const QString xml = serializeMaterialXGraph(nodes, udimSet_);
    return xml.trimmed().isEmpty() ? fallbackDefaultDocument() : xml;
}

void MaterialNetworkGraphView::writeModel(bool emitEdited) { writeXmlToMaterial(serializeGraph(), emitEdited); }

void MaterialNetworkGraphView::showEvent(QShowEvent* event) {
    QGraphicsView::showEvent(event);
    // Always reframe when the MaterialX canvas becomes visible. Previously this
    // only ran when pendingFrame_ was already set, so dive-into left nodes stuck
    // on the left edge of the dock after the first tiny layout pass.
    requestFrame();
}

void MaterialNetworkGraphView::resizeEvent(QResizeEvent* event) {
    QGraphicsView::resizeEvent(event);
    requestFrame();
}

void MaterialNetworkGraphView::requestFrame() { scheduleFrameGraph(); }

void MaterialNetworkGraphView::scheduleFrameGraph() {
    pendingFrame_ = true;
    auto attemptFrame = [this]() {
        if (!pendingFrame_) return;
        if (!isVisible() || width() < 80 || height() < 60) return;
        frameGraph();
    };
    // Dock/tabify often settles over a few events — reframe now and again shortly after.
    QTimer::singleShot(0, this, attemptFrame);
    QTimer::singleShot(33, this, attemptFrame);
    QTimer::singleShot(100, this, attemptFrame);
}

void MaterialNetworkGraphView::frameGraph() {
    if (width() < 80 || height() < 60) {
        pendingFrame_ = true;
        return;
    }
    pendingFrame_ = false;
    graphScene_->setSceneRect(-8000, -8000, 16000, 16000);

    // Bounds from node bodies only — long wires / port pads used to skew fitInView
    // so the graph sat on the left of a huge empty scene.
    QRectF bounds;
    for (QGraphicsItem* item : graphScene_->items()) {
        if (item->type() == MaterialNetworkNodeItem::Type) bounds |= item->sceneBoundingRect();
    }
    if (bounds.isEmpty()) bounds = graphScene_->itemsBoundingRect();

    const QGraphicsView::ViewportAnchor savedAnchor = transformationAnchor();
    setTransformationAnchor(QGraphicsView::AnchorViewCenter);
    resetTransform();
    if (bounds.isEmpty()) {
        centerOn(0.0, 0.0);
        setTransformationAnchor(savedAnchor);
        return;
    }
    const QRectF padded = bounds.adjusted(-140.0, -100.0, 140.0, 100.0);
    fitInView(padded, Qt::KeepAspectRatio);
    const double scaleValue = transform().m11();
    if (scaleValue > 1.15 || scaleValue < 0.55) {
        resetTransform();
        const double clamped = std::clamp(scaleValue, 0.55, 1.15);
        scale(clamped, clamped);
    }
    centerOn(bounds.center());
    setTransformationAnchor(savedAnchor);
}

void MaterialNetworkGraphView::wheelEvent(QWheelEvent* event) {
    const qreal factor = graphicsViewWheelZoomFactor(event);
    zoomGraphicsViewAtCursor(this, factor, event->globalPosition(),
                             panning_ ? &panScenePoint_ : nullptr, 0.16, 4.0);
    event->accept();
}

void MaterialNetworkGraphView::keyPressEvent(QKeyEvent* event) {
    if (event->matches(QKeySequence::Copy)) {
        copySelectedNodes();
        event->accept();
        return;
    }
    if (event->matches(QKeySequence::Paste)) {
        pasteNodes();
        event->accept();
        return;
    }
    switch (event->key()) {
        case Qt::Key_Space:
            if (!event->isAutoRepeat()) {
                spacePressed_ = true;
                if (!panning_) viewport()->setCursor(Qt::OpenHandCursor);
            }
            event->accept();
            return;
        case Qt::Key_Tab: {
            const QPoint position =
                viewport()->rect().contains(lastMousePoint_) ? lastMousePoint_ : viewport()->rect().center();
            showAddNodeMenu(position);
            event->accept();
            return;
        }
        case Qt::Key_F:
            frameGraph();
            event->accept();
            return;
        case Qt::Key_Up:
            emit upRequested();
            event->accept();
            return;
        case Qt::Key_Delete:
        case Qt::Key_Backspace:
            deleteSelectedNodes();
            event->accept();
            return;
        default:
            break;
    }
    QGraphicsView::keyPressEvent(event);
}

void MaterialNetworkGraphView::keyReleaseEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Space && !event->isAutoRepeat()) {
        spacePressed_ = false;
        if (!panning_) viewport()->unsetCursor();
        event->accept();
        return;
    }
    QGraphicsView::keyReleaseEvent(event);
}

void MaterialNetworkGraphView::contextMenuEvent(QContextMenuEvent* event) {
    lastMousePoint_ = event->pos();

    if (MaterialWireItem* wire = wireItemAt(this, event->pos())) {
        QMenu menu(this);
        const QString target = wire->targetNodeName();
        const QString input = wire->inputName();
        menu.addAction("Disconnect", this, [this, target, input] { disconnectInput(target, input); });
        menu.exec(event->globalPos());
        event->accept();
        return;
    }

    if (!nodeItemAt(this, event->pos())) {
        showAddNodeMenu(event->pos());
        event->accept();
        return;
    }
    QGraphicsView::contextMenuEvent(event);
}

void MaterialNetworkGraphView::showAddNodeMenu(const QPoint& viewPosition) {
    if (!materialNode_) {
        emit statusMessage("Select a Material node in the Scene Network");
        return;
    }
    if (catalogCache().isEmpty()) {
        emit statusMessage("MaterialX node catalog is empty");
        return;
    }
    pendingCreateScenePos_ = mapToScene(viewPosition);
    if (createMenu_) createMenu_->popupAt(viewport()->mapToGlobal(viewPosition));
}

void MaterialNetworkGraphView::onCreateMenuChosen(const QString& category, const QString& type) {
    addNode(category, type, pendingCreateScenePos_);
}

void MaterialNetworkGraphView::addNode(const QString& category, const QString& type, QPointF scenePosition) {
    if (!materialNode_ || !isKnownMaterialXCategory(category)) return;
    MtlxNode node;
    node.category = category;
    node.type = type.isEmpty() ? defaultTypeForCategory(category) : type;
    node.name = uniqueNodeName(category == "surfacematerial" ? "surfacematerial" : category);
    node.layout = scenePosition / kLayoutScale;
    node.inputs = defaultInputsForCategory(category, node.type);
    graphNodes_.push_back(node);
    writeModel(true);
    rebuild();
    emit statusMessage("Added " + category + " (" + node.type + ")");
}

void MaterialNetworkGraphView::connectNodes(const QString& sourceName, const QString& targetName, int inputIndex) {
    if (sourceName.isEmpty() || targetName.isEmpty() || sourceName == targetName) return;
    MtlxNode* target = findModelNode(targetName);
    const MtlxNode* source = findModelNode(sourceName);
    if (!target || !source || inputIndex < 0 || inputIndex >= target->inputs.size()) return;
    if (!isConnectableInput(target->inputs[inputIndex].name, target->inputs[inputIndex].type)) return;
    if (!materialXTypesConnectable(source->type, target->inputs[inputIndex].type)) {
        emit statusMessage(QString("Cannot connect %1 (%2) → %3.%4 (%5)")
                               .arg(sourceName, source->type, targetName, target->inputs[inputIndex].name,
                                    target->inputs[inputIndex].type));
        return;
    }
    target->inputs[inputIndex].nodename = sourceName;
    target->inputs[inputIndex].value.clear();
    writeModel(true);
    // Defer rebuild: connecting during mouseRelease while Parameters/scene still
    // walk the old QGraphicsItems was crashing on Windows (same class of bug as
    // triplanar file-pick rebuild-on-signal).
    preservedSelection_ = targetName;
    QTimer::singleShot(0, this, [this] {
        if (!materialNode_) return;
        rebuild();
    });
    emit statusMessage(QString("Connected %1 to %2.%3").arg(sourceName, targetName, target->inputs[inputIndex].name));
}

void MaterialNetworkGraphView::disconnectInput(const QString& targetName, const QString& inputName) {
    MtlxNode* target = findModelNode(targetName);
    if (!target || inputName.isEmpty()) return;
    for (MtlxInput& input : target->inputs) {
        if (input.name != inputName) continue;
        if (input.nodename.isEmpty()) return;
        input.nodename.clear();
        writeModel(true);
        preservedSelection_ = targetName;
        QTimer::singleShot(0, this, [this] {
            if (!materialNode_) return;
            rebuild();
        });
        emit statusMessage(QString("Disconnected %1.%2").arg(targetName, inputName));
        return;
    }
}

void MaterialNetworkGraphView::deleteSelectedNodes() {
    if (!materialNode_) return;
    QStringList toDelete;
    bool skippedSurface = false;
    for (QGraphicsItem* item : graphScene_->selectedItems()) {
        if (auto* nodeItem = qgraphicsitem_cast<MaterialNetworkNodeItem*>(item)) {
            if (nodeItem->nodeName() == "surface" && nodeItem->category() == "surfacematerial") {
                skippedSurface = true;
                continue;
            }
            toDelete << nodeItem->nodeName();
        }
    }
    toDelete.removeDuplicates();
    if (toDelete.isEmpty()) {
        if (skippedSurface) emit statusMessage("The terminal surface material is kept");
        return;
    }

    graphNodes_.erase(std::remove_if(graphNodes_.begin(), graphNodes_.end(), [&toDelete](const MtlxNode& node) {
                          return toDelete.contains(node.name);
                      }),
                      graphNodes_.end());
    for (MtlxNode& node : graphNodes_) {
        for (MtlxInput& input : node.inputs) {
            if (toDelete.contains(input.nodename)) input.nodename.clear();
        }
    }
    writeModel(true);
    rebuild();
    emit statusMessage("Deleted " + toDelete.join(", "));
}

namespace {
constexpr const char* kMtlxClipboardMime = "application/x-bob-render-mtlx-nodes";
}

void MaterialNetworkGraphView::copySelectedNodes() {
    if (!materialNode_) return;
    QStringList names;
    for (QGraphicsItem* item : graphScene_->selectedItems()) {
        if (auto* nodeItem = qgraphicsitem_cast<MaterialNetworkNodeItem*>(item)) names << nodeItem->nodeName();
    }
    names.removeDuplicates();
    if (names.isEmpty()) {
        emit statusMessage("Nothing to copy");
        return;
    }

    QJsonArray nodesArray;
    const QSet<QString> selected(names.begin(), names.end());
    for (const QString& name : names) {
        const MtlxNode* node = findModelNode(name);
        if (!node) continue;
        QJsonObject nodeJson;
        nodeJson["name"] = node->name;
        nodeJson["category"] = node->category;
        nodeJson["type"] = node->type;
        nodeJson["x"] = node->layout.x();
        nodeJson["y"] = node->layout.y();
        QJsonArray inputsArray;
        for (const MtlxInput& input : node->inputs) {
            QJsonObject inputJson;
            inputJson["name"] = input.name;
            inputJson["type"] = input.type;
            inputJson["value"] = input.value;
            // Keep only connections fully inside the selection.
            inputJson["nodename"] = selected.contains(input.nodename) ? input.nodename : QString();
            inputsArray.append(inputJson);
        }
        nodeJson["inputs"] = inputsArray;
        nodesArray.append(nodeJson);
    }

    QJsonObject root;
    root["format"] = QStringLiteral("bob-render-mtlx-nodes");
    root["version"] = 1;
    root["nodes"] = nodesArray;
    const QByteArray bytes = QJsonDocument(root).toJson(QJsonDocument::Compact);
    auto* mime = new QMimeData;
    mime->setData(kMtlxClipboardMime, bytes);
    mime->setText(QString::fromUtf8(bytes));
    QApplication::clipboard()->setMimeData(mime);
    emit statusMessage(QString("Copied %1 MaterialX node%2")
                           .arg(nodesArray.size())
                           .arg(nodesArray.size() == 1 ? "" : "s"));
}

void MaterialNetworkGraphView::pasteNodes() {
    if (!materialNode_) return;
    const QMimeData* mime = QApplication::clipboard()->mimeData();
    if (!mime) return;

    QByteArray bytes;
    if (mime->hasFormat(kMtlxClipboardMime))
        bytes = mime->data(kMtlxClipboardMime);
    else if (mime->hasText())
        bytes = mime->text().toUtf8();
    if (bytes.isEmpty()) {
        emit statusMessage("Clipboard has no MaterialX nodes");
        return;
    }

    const QJsonDocument doc = QJsonDocument::fromJson(bytes);
    if (!doc.isObject()) {
        emit statusMessage("Clipboard is not MaterialX node data");
        return;
    }
    const QJsonObject root = doc.object();
    if (root.value("format").toString() != QLatin1String("bob-render-mtlx-nodes")) {
        emit statusMessage("Clipboard is not MaterialX node data");
        return;
    }
    const QJsonArray nodesArray = root.value("nodes").toArray();
    if (nodesArray.isEmpty()) {
        emit statusMessage("Clipboard has no MaterialX nodes");
        return;
    }

    QPointF cursorLayout = lastMousePoint_;
    const QPoint viewPos = mapFromGlobal(QCursor::pos());
    if (viewport()->rect().contains(viewPos)) cursorLayout = mapToScene(viewPos);
    cursorLayout /= kLayoutScale;

    QPointF minLayout(nodesArray[0].toObject().value("x").toDouble(),
                      nodesArray[0].toObject().value("y").toDouble());
    QPointF maxLayout = minLayout;
    for (const QJsonValue& value : nodesArray) {
        const QJsonObject o = value.toObject();
        const QPointF p(o.value("x").toDouble(), o.value("y").toDouble());
        minLayout.setX(std::min(minLayout.x(), p.x()));
        minLayout.setY(std::min(minLayout.y(), p.y()));
        maxLayout.setX(std::max(maxLayout.x(), p.x()));
        maxLayout.setY(std::max(maxLayout.y(), p.y()));
    }
    const QPointF center((minLayout.x() + maxLayout.x()) * 0.5, (minLayout.y() + maxLayout.y()) * 0.5);
    const QPointF delta = cursorLayout - center;

    QHash<QString, QString> nameMap;
    QStringList createdNames;
    for (const QJsonValue& value : nodesArray) {
        const QJsonObject nodeJson = value.toObject();
        const QString category = nodeJson.value("category").toString();
        if (!isKnownMaterialXCategory(category)) continue;
        // Never paste a second terminal "surface" container as the fixed name.
        const QString oldName = nodeJson.value("name").toString();
        QString baseName = oldName;
        if (category == "surfacematerial" && oldName == "surface") baseName = "surfacematerial";
        MtlxNode node;
        node.category = category;
        node.type = nodeJson.value("type").toString();
        if (node.type.isEmpty()) node.type = defaultTypeForCategory(category);
        node.name = uniqueNodeName(baseName.isEmpty() ? category : baseName);
        node.layout = QPointF(nodeJson.value("x").toDouble(), nodeJson.value("y").toDouble()) + delta;
        const QJsonArray inputsArray = nodeJson.value("inputs").toArray();
        for (const QJsonValue& inputValue : inputsArray) {
            const QJsonObject inputJson = inputValue.toObject();
            MtlxInput input;
            input.name = inputJson.value("name").toString();
            input.type = inputJson.value("type").toString();
            input.value = inputJson.value("value").toString();
            input.nodename = inputJson.value("nodename").toString();
            if (!input.name.isEmpty()) node.inputs.append(input);
        }
        if (node.inputs.isEmpty()) node.inputs = defaultInputsForCategory(category, node.type);
        nameMap.insert(oldName, node.name);
        createdNames << node.name;
        graphNodes_.push_back(node);
    }

    if (createdNames.isEmpty()) {
        emit statusMessage("Paste failed");
        return;
    }

    for (MtlxNode& node : graphNodes_) {
        if (!createdNames.contains(node.name)) continue;
        for (MtlxInput& input : node.inputs) {
            if (input.nodename.isEmpty()) continue;
            input.nodename = nameMap.value(input.nodename);
        }
    }

    writeModel(true);
    rebuild();
    preservedSelection_ = createdNames.isEmpty() ? QString() : createdNames.first();
    emit statusMessage(QString("Pasted %1 MaterialX node%2")
                           .arg(createdNames.size())
                           .arg(createdNames.size() == 1 ? "" : "s"));
}

void MaterialNetworkGraphView::mousePressEvent(QMouseEvent* event) {
    lastMousePoint_ = event->pos();
    mousePressPoint_ = event->pos();
    mouseMovedSincePress_ = false;
    clickImageNode_.clear();

    if (shouldBeginPan(event)) {
        beginPan(event->globalPosition());
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton) {
        const QPointF scenePos = mapToScene(event->pos());
        // Tight radius: MaterialX ports sit on the node edge; a wide scan used to
        // steal body clicks for wire-drag (fist cursor) instead of selection.
        constexpr qreal kTightPort = kPortRadius + 4.0;

        if (MaterialNetworkNodeItem* item = nodeItemAt(this, event->pos())) {
            const bool onBody = item->containsBody(scenePos);
            const bool onOutput = item->hitOutputPort(scenePos, kTightPort);
            const int onInput = item->hitInputPort(scenePos, kTightPort);

            if (onOutput) {
                beginWire(item->nodeName(), item->outputPortScene());
                event->accept();
                return;
            }
            if (onInput >= 0 && !onBody) {
                const int modelInput = item->inputModelIndex(onInput);
                if (MtlxNode* target = findModelNode(item->nodeName());
                    target && modelInput >= 0 && modelInput < target->inputs.size()) {
                    const QString existing = target->inputs[modelInput].nodename;
                    if (!existing.isEmpty()) {
                        target->inputs[modelInput].nodename.clear();
                        writeModel(false);
                        rebuild();
                        if (MaterialNetworkNodeItem* sourceItem = nodeItemByName(graphScene_, existing)) {
                            beginWire(existing, sourceItem->outputPortScene());
                            updateWire(scenePos);
                            event->accept();
                            return;
                        }
                    }
                }
            }
            if (onInput >= 0 && onBody) {
                // Click on the port knob itself (inside body margin) → rewire.
                if (QLineF(scenePos, item->inputPortScene(onInput)).length() <= kTightPort) {
                    const int modelInput = item->inputModelIndex(onInput);
                    if (MtlxNode* target = findModelNode(item->nodeName());
                        target && modelInput >= 0 && modelInput < target->inputs.size()) {
                        const QString existing = target->inputs[modelInput].nodename;
                        if (!existing.isEmpty()) {
                            target->inputs[modelInput].nodename.clear();
                            writeModel(false);
                            rebuild();
                            if (MaterialNetworkNodeItem* sourceItem = nodeItemByName(graphScene_, existing)) {
                                beginWire(existing, sourceItem->outputPortScene());
                                updateWire(scenePos);
                                event->accept();
                                return;
                            }
                        }
                    }
                }
            }

            // Body click: select immediately so Parameters update (Scene Network style).
            {
                const QSignalBlocker blocker(graphScene_);
                if (!(event->modifiers() & Qt::ShiftModifier)) graphScene_->clearSelection();
                item->setSelected(true);
            }
            if (item->category() == "image" || item->category() == "tiledimage")
                clickImageNode_ = item->nodeName();
            QGraphicsView::mousePressEvent(event);
            // Always push Parameters — Qt skips selectionChanged when the item
            // was already selected, which left the panel stuck on the container.
            emitSelectionChanged();
            return;
        }

        // Empty canvas near a port — start / pull a wire.
        if (MaterialNetworkNodeItem* source = outputPortAt(graphScene_, scenePos, kPortHitRadius)) {
            beginWire(source->nodeName(), source->outputPortScene());
            event->accept();
            return;
        }
        if (const InputHit hit = inputPortAt(graphScene_, scenePos, kPortHitRadius); hit.item) {
            const int modelInput = hit.item->inputModelIndex(hit.inputIndex);
            if (MtlxNode* target = findModelNode(hit.item->nodeName());
                target && modelInput >= 0 && modelInput < target->inputs.size()) {
                const QString existing = target->inputs[modelInput].nodename;
                if (!existing.isEmpty()) {
                    target->inputs[modelInput].nodename.clear();
                    writeModel(false);
                    rebuild();
                    if (MaterialNetworkNodeItem* sourceItem = nodeItemByName(graphScene_, existing)) {
                        beginWire(existing, sourceItem->outputPortScene());
                        updateWire(scenePos);
                        event->accept();
                        return;
                    }
                }
            }
        }
    }

    QGraphicsView::mousePressEvent(event);
}

void MaterialNetworkGraphView::mouseMoveEvent(QMouseEvent* event) {
    lastMousePoint_ = event->pos();
    if (QLineF(event->pos(), mousePressPoint_).length() > 4.0) mouseMovedSincePress_ = true;

    if (panning_) {
        updatePan(event->globalPosition());
        event->accept();
        return;
    }
    if (wiring_) {
        updateWire(mapToScene(event->pos()));
        event->accept();
        return;
    }

    // Hover tip: show the MaterialX data type name on the port under the cursor.
    const QPointF scenePos = mapToScene(event->pos());
    QString tip;
    if (MaterialNetworkNodeItem* source = outputPortAt(graphScene_, scenePos, kPortHitRadius)) {
        tip = source->typeName().isEmpty() ? QStringLiteral("output") : source->typeName();
    } else if (const InputHit hit = inputPortAt(graphScene_, scenePos, kPortHitRadius); hit.item) {
        tip = hit.item->inputPortType(hit.inputIndex);
        if (tip.isEmpty()) tip = hit.item->inputPortName(hit.inputIndex);
    }
    if (!tip.isEmpty()) {
        QToolTip::showText(event->globalPosition().toPoint(), tip, this);
    } else {
        QToolTip::hideText();
    }

    QGraphicsView::mouseMoveEvent(event);
}

void MaterialNetworkGraphView::mouseReleaseEvent(QMouseEvent* event) {
    lastMousePoint_ = event->pos();
    if (panning_) {
        endPan();
        event->accept();
        return;
    }
    if (wiring_) {
        endWire(event->pos());
        event->accept();
        return;
    }

    QGraphicsView::mouseReleaseEvent(event);
    syncNodePositions();

    if (event->button() == Qt::LeftButton && !mouseMovedSincePress_ && !clickImageNode_.isEmpty()) {
        MaterialNetworkNodeItem* item = nodeItemAt(this, event->pos());
        if (item && item->nodeName() == clickImageNode_) chooseTexture(clickImageNode_);
    }
    clickImageNode_.clear();
}

void MaterialNetworkGraphView::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && openTextureDialogAt(event->pos())) {
        event->accept();
        return;
    }
    QGraphicsView::mouseDoubleClickEvent(event);
}

bool MaterialNetworkGraphView::openTextureDialogAt(const QPoint& viewPosition) {
    MaterialNetworkNodeItem* item = nodeItemAt(this, viewPosition);
    if (!item || (item->category() != "image" && item->category() != "tiledimage")) return false;
    graphScene_->clearSelection();
    item->setSelected(true);
    chooseTexture(item->nodeName());
    return true;
}

QString MaterialNetworkGraphView::applyUdimFilename(const QString& path) {
    QString pattern;
    std::vector<int> tiles;
    if (!resolveUdimPattern(path, QString(), pattern, tiles)) return path;
    udimSet_.clear();
    for (int udim : tiles) udimSet_.push_back(udim);
    if (udimSet_.size() > 1) {
        emit statusMessage(QString("UDIM: %1 tile(s) [%2…%3]")
                               .arg(udimSet_.size())
                               .arg(udimSet_.first())
                               .arg(udimSet_.last()));
    }
    return pattern;
}

void MaterialNetworkGraphView::refreshUdimSetFromFilenames() {
    // Rebuild udimset from current image filenames (MaterialX geominfo source of truth).
    QVector<int> merged;
    for (const MtlxNode& node : graphNodes_) {
        if (node.category != "image" && node.category != "tiledimage") continue;
        for (const MtlxInput& input : node.inputs) {
            if (input.name != "file" || input.value.isEmpty() || !input.nodename.isEmpty()) continue;
            QString pattern;
            std::vector<int> tiles;
            if (!resolveUdimPattern(input.value, QString(), pattern, tiles)) continue;
            for (int udim : tiles) {
                if (!merged.contains(udim)) merged.push_back(udim);
            }
        }
    }
    // Keep authoring-time udimset when files are missing on this machine.
    if (merged.isEmpty() && !udimSet_.isEmpty()) return;
    std::sort(merged.begin(), merged.end());
    udimSet_ = merged;
}

void MaterialNetworkGraphView::chooseTexture(const QString& nodeName) {
    MtlxNode* node = findModelNode(nodeName);
    if (!node || (node->category != "image" && node->category != "tiledimage")) return;

    int fileInput = -1;
    for (int i = 0; i < node->inputs.size(); ++i) {
        if (node->inputs[i].name == "file") {
            fileInput = i;
            break;
        }
    }
    if (fileInput < 0) {
        node->inputs.push_back({"file", "filename", {}, {}});
        fileInput = node->inputs.size() - 1;
    }

    const QString filter = "Images (*.png *.jpg *.jpeg *.exr *.hdr *.tx *.tif *.tiff *.bmp *.webp);;All Files (*)";
    const auto picked = TextureFileDialog::getOpenTexture(
        this, "Choose texture for " + nodeName, node->inputs[fileInput].value, filter);
    if (picked.path.isEmpty()) return;

    // UDIM sequence → keep unresolved <UDIM> + geominfo udimset.
    // $F sequence → store the token path as authored.
    if (picked.token == SequenceTokenKind::Udim || picked.path.contains(QLatin1String("<UDIM>"))) {
        node->inputs[fileInput].value = applyUdimFilename(picked.path);
    } else {
        node->inputs[fileInput].value = picked.path;
    }
    node->inputs[fileInput].nodename.clear();
    writeModel(true);
    rebuild();
    emit statusMessage(QString("%1 file set to %2").arg(nodeName, QFileInfo(node->inputs[fileInput].value).fileName()));
}

void MaterialNetworkGraphView::updateWiresLive() {
    if (!graphScene_) return;
    for (QGraphicsItem* item : graphScene_->items()) {
        auto* wire = qgraphicsitem_cast<MaterialWireItem*>(item);
        if (!wire) continue;
        MaterialNetworkNodeItem* source = nodeItemByName(graphScene_, wire->sourceNodeName());
        MaterialNetworkNodeItem* target = nodeItemByName(graphScene_, wire->targetNodeName());
        if (!source || !target) continue;
        const int port = target->inputPortIndexByName(wire->inputName());
        if (port < 0) continue;
        wire->setWirePath(makeWirePath(source->outputPortScene(), target->inputPortScene(port)));
    }
}

void MaterialNetworkGraphView::persistLayoutQuietly() {
    if (!materialNode_) return;
    ensureMtlxParameter();
    const QString xml = serializeGraph();
    // Write layout into the stored document without dirtying cook / emitting signals.
    if (Parameter* parameter = materialNode_->findParameter("mtlx")) {
        if (parameter->value.toString() == xml) return;
        parameter->value = xml;
    }
    emit materialLayoutChanged();
}

void MaterialNetworkGraphView::syncNodePositions() {
    bool changed = false;
    for (QGraphicsItem* item : graphScene_->items()) {
        auto* nodeItem = qgraphicsitem_cast<MaterialNetworkNodeItem*>(item);
        if (!nodeItem) continue;
        MtlxNode* node = findModelNode(nodeItem->nodeName());
        if (!node) continue;
        const QPointF layout = nodeItem->pos() / kLayoutScale;
        if (QLineF(node->layout, layout).length() > 0.001) {
            node->layout = layout;
            changed = true;
        }
    }
    if (!changed) return;
    // Persist xpos/ypos only — no cook, no scene rebuild (matches Scene Network).
    persistLayoutQuietly();
}

bool MaterialNetworkGraphView::shouldBeginPan(const QMouseEvent* event) const {
    // Match the Scene Network (Houdini-style):
    //   MMB drag       — pan
    //   Alt+LMB drag   — pan
    //   Space+LMB drag — pan
    if (event->button() == Qt::MiddleButton) return true;
    if (event->button() == Qt::LeftButton && (event->modifiers() & Qt::AltModifier)) return true;
    if (event->button() == Qt::LeftButton && spacePressed_) return true;
    return false;
}

void MaterialNetworkGraphView::beginPan(const QPointF& globalPos) {
    panning_ = true;
    savedDragMode_ = dragMode();
    savedAnchor_ = transformationAnchor();
    setDragMode(QGraphicsView::NoDrag);
    setTransformationAnchor(QGraphicsView::NoAnchor);
    setResizeAnchor(QGraphicsView::NoAnchor);
    viewport()->grabMouse();
    viewport()->setCursor(Qt::ClosedHandCursor);
    setFocus(Qt::MouseFocusReason);
    panScenePoint_ = mapToScene(graphicsViewViewportPos(this, globalPos));
}

void MaterialNetworkGraphView::updatePan(const QPointF& globalPos) {
    if (!panning_) return;
    glueGraphicsViewPan(this, panScenePoint_, globalPos);
}

void MaterialNetworkGraphView::endPan() {
    if (!panning_) return;
    panning_ = false;
    if (QWidget::mouseGrabber() == viewport()) viewport()->releaseMouse();
    setDragMode(savedDragMode_);
    setTransformationAnchor(savedAnchor_);
    viewport()->unsetCursor();
}

void MaterialNetworkGraphView::beginWire(const QString& sourceName, QPointF sourcePosition) {
    wiring_ = true;
    wireSourceNode_ = sourceName;
    wireSourcePosition_ = sourcePosition;
    setDragMode(QGraphicsView::NoDrag);
    QColor wireColor = theme::wireActive();
    if (const MtlxNode* source = findModelNode(sourceName))
        wireColor = theme::colorForMaterialXType(source->type);
    previewWire_ = graphScene_->addPath(makeWirePath(sourcePosition, sourcePosition),
                                        QPen(wireColor, 1.8, Qt::DashLine, Qt::RoundCap, Qt::RoundJoin));
    previewWire_->setZValue(4.0);
}

void MaterialNetworkGraphView::updateWire(QPointF scenePosition) {
    if (previewWire_) previewWire_->setPath(makeWirePath(wireSourcePosition_, scenePosition));
}

void MaterialNetworkGraphView::endWire(const QPoint& viewPosition) {
    const QString sourceName = wireSourceNode_;
    const QPointF scenePosition = mapToScene(viewPosition);
    InputHit hit = inputPortAt(graphScene_, scenePosition, kPortSnapRadius);
    if (previewWire_) {
        graphScene_->removeItem(previewWire_);
        delete previewWire_;
        previewWire_ = nullptr;
    }
    wiring_ = false;
    setDragMode(QGraphicsView::RubberBandDrag);
    wireSourceNode_.clear();

    if (sourceName.isEmpty()) return;
    if (!hit.item) {
        // Soft snap: drop on a node body and pick the best type-compatible input
        // (Arnold-style: color patterns prefer base_color / Diffuse Color, not base weight).
        if (MaterialNetworkNodeItem* item = nodeItemAt(this, viewPosition)) {
            if (item->nodeName() != sourceName && item->inputCount() > 0) {
                const MtlxNode* source = findModelNode(sourceName);
                const QString sourceType = source ? source->type : QString();
                int bestPort = -1;
                int bestScore = -1;
                int nearPort = item->hitInputPort(scenePosition, kPortSnapRadius * 2.0);
                for (int port = 0; port < item->inputCount(); ++port) {
                    const int modelInput = item->inputModelIndex(port);
                    const MtlxNode* target = findModelNode(item->nodeName());
                    if (!target || modelInput < 0 || modelInput >= target->inputs.size()) continue;
                    const MtlxInput& input = target->inputs[modelInput];
                    if (!isConnectableInput(input.name, input.type)) continue;
                    int score = connectionScore(sourceType, input.name, input.type, !input.nodename.isEmpty());
                    if (score < 0) continue;
                    if (port == nearPort) score += 15;
                    if (score > bestScore) {
                        bestScore = score;
                        bestPort = port;
                    }
                }
                if (bestPort >= 0) hit = {item, bestPort};
            }
        }
    }
    if (!hit.item || hit.item->nodeName() == sourceName) return;
    const int modelInput = hit.item->inputModelIndex(hit.inputIndex);
    connectNodes(sourceName, hit.item->nodeName(), modelInput);
}

void MaterialNetworkGraphView::drawBackground(QPainter* painter, const QRectF& rect) {
    painter->fillRect(rect, theme::gridDark());

    const qreal step = 40.0;
    QPen minor(theme::gridLine(), 0.0);
    painter->setPen(minor);
    const qreal left = std::floor(rect.left() / step) * step;
    const qreal top = std::floor(rect.top() / step) * step;
    for (qreal x = left; x < rect.right(); x += step)
        painter->drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()));
    for (qreal y = top; y < rect.bottom(); y += step)
        painter->drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y));

    QPen major(theme::gridLine().lighter(130), 0.0);
    painter->setPen(major);
    const qreal bigStep = step * 5.0;
    const qreal bigLeft = std::floor(rect.left() / bigStep) * bigStep;
    const qreal bigTop = std::floor(rect.top() / bigStep) * bigStep;
    for (qreal x = bigLeft; x < rect.right(); x += bigStep)
        painter->drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()));
    for (qreal y = bigTop; y < rect.bottom(); y += bigStep)
        painter->drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y));
}

void MaterialNetworkGraphView::drawForeground(QPainter* painter, const QRectF& rect) {
    QGraphicsView::drawForeground(painter, rect);
    if (materialNode_) return;
    painter->resetTransform();
    QFont font = painter->font();
    font.setPointSizeF(11.0);
    painter->setFont(font);
    painter->setPen(theme::textDim());
    painter->drawText(QRect(0, 0, width(), height()), Qt::AlignCenter,
                      "Double-click a material container to edit its MaterialX graph");
}

void MaterialNetworkGraphView::emitSelectionChanged() { emit selectionChanged(); }

QString MaterialNetworkGraphView::selectedNodeName() const {
    QString name;
    for (QGraphicsItem* item : graphScene_->selectedItems()) {
        if (auto* nodeItem = qgraphicsitem_cast<MaterialNetworkNodeItem*>(item)) {
            if (!name.isEmpty() && name != nodeItem->nodeName()) return {};
            name = nodeItem->nodeName();
        }
    }
    return name;
}

const MaterialNetworkGraphView::MtlxNode* MaterialNetworkGraphView::selectedNode() const {
    return findModelNode(selectedNodeName());
}

void MaterialNetworkGraphView::selectNodeByName(const QString& name) {
    graphScene_->clearSelection();
    if (name.isEmpty()) {
        emitSelectionChanged();
        return;
    }
    if (MaterialNetworkNodeItem* item = nodeItemByName(graphScene_, name)) {
        item->setSelected(true);
        ensureVisible(item);
    }
    emitSelectionChanged();
}

bool MaterialNetworkGraphView::renameNode(const QString& oldName, const QString& newName) {
    const QString trimmed = newName.trimmed();
    if (oldName.isEmpty() || trimmed.isEmpty() || trimmed == oldName) return false;
    if (!findModelNode(oldName)) return false;
    if (findModelNode(trimmed)) {
        emit statusMessage("Node name already exists: " + trimmed);
        return false;
    }
    // MaterialX names should be simple identifiers.
    static const QRegularExpression validName(QStringLiteral("^[A-Za-z_][A-Za-z0-9_]*$"));
    if (!validName.match(trimmed).hasMatch()) {
        emit statusMessage("Invalid MaterialX node name");
        return false;
    }

    for (MtlxNode& node : graphNodes_) {
        if (node.name == oldName) node.name = trimmed;
        for (MtlxInput& input : node.inputs) {
            if (input.nodename == oldName) input.nodename = trimmed;
        }
    }
    preservedSelection_ = trimmed;
    writeModel(true);
    rebuild();
    emit statusMessage(QString("Renamed %1 → %2").arg(oldName, trimmed));
    return true;
}

bool MaterialNetworkGraphView::setInputValue(const QString& nodeName, const QString& inputName,
                                             const QString& value) {
    MtlxNode* node = findModelNode(nodeName);
    if (!node) return false;
    for (MtlxInput& input : node->inputs) {
        if (input.name != inputName) continue;
        const bool wasConnected = !input.nodename.isEmpty();
        QString stored = value;
        if (input.type == "filename" || inputName == "file" || inputName.startsWith("file"))
            stored = applyUdimFilename(value);
        if (input.value == stored && !wasConnected) return true;
        input.value = stored;
        input.nodename.clear();

        // Arnold Triplanar: shared Input copies to all axes unless Input Per Axis is on.
        bool refreshParamsOnly = false;
        if (node->category == "triplanarprojection") {
            auto findIn = [&](const QString& n) -> MtlxInput* {
                for (MtlxInput& in : node->inputs) {
                    if (in.name == n) return &in;
                }
                return nullptr;
            };
            MtlxInput* perAxis = findIn("input_per_axis");
            const bool separate = perAxis && (perAxis->value == "true" || perAxis->value == "1");
            if (inputName == "file" && !separate) {
                for (const char* axis : {"filex", "filey", "filez"}) {
                    if (MtlxInput* axisIn = findIn(QLatin1String(axis))) {
                        axisIn->value = stored;
                        axisIn->nodename.clear();
                    }
                }
            } else if (inputName == "input_per_axis") {
                refreshParamsOnly = true;
                if (separate) {
                    // Enabling per-axis: seed empty axis slots from shared file.
                    QString seed;
                    if (MtlxInput* file = findIn("file")) seed = file->value;
                    if (seed.isEmpty()) {
                        if (MtlxInput* fx = findIn("filex")) seed = fx->value;
                    }
                    if (!seed.isEmpty()) {
                        for (const char* axis : {"filex", "filey", "filez"}) {
                            if (MtlxInput* axisIn = findIn(QLatin1String(axis))) {
                                if (axisIn->value.isEmpty()) axisIn->value = seed;
                            }
                        }
                    }
                } else {
                    // Disabling: collapse back to shared file (prefer file, else filex).
                    QString seed;
                    if (MtlxInput* file = findIn("file")) seed = file->value;
                    if (seed.isEmpty()) {
                        if (MtlxInput* fx = findIn("filex")) seed = fx->value;
                    }
                    if (!seed.isEmpty()) {
                        if (MtlxInput* file = findIn("file")) file->value = seed;
                        for (const char* axis : {"filex", "filey", "filez"}) {
                            if (MtlxInput* axisIn = findIn(QLatin1String(axis))) axisIn->value = seed;
                        }
                    }
                }
            }
        }

        writeModel(true);
        // Avoid graph rebuild while Parameters editors are still in their signal stack
        // (browse/editFinished) — that was crashing triplanar file picks.
        // Rebuild only when wires or image-node subtitles change.
        const bool imageSubtitle =
            (node->category == "image" || node->category == "tiledimage") && input.type == "filename";
        if (wasConnected || imageSubtitle) {
            preservedSelection_ = nodeName;
            rebuild();
        }
        // input_per_axis visibility refresh is deferred by MainWindow (QTimer),
        // not here — rebuilding Parameters inside QCheckBox::toggled crashes.
        Q_UNUSED(refreshParamsOnly);
        return true;
    }
    return false;
}

bool MaterialNetworkGraphView::setNodeType(const QString& nodeName, const QString& type) {
    MtlxNode* node = findModelNode(nodeName);
    if (!node || type.isEmpty() || type == node->type) return false;

    const MaterialXNodeCatalogEntry* entry = findCatalogEntry(node->category);
    if (entry && !entry->typeVariants.isEmpty() && !entry->typeVariants.contains(type)) {
        emit statusMessage("Unsupported type for " + node->category + ": " + type);
        return false;
    }

    QHash<QString, MtlxInput> previous;
    previous.reserve(node->inputs.size());
    for (const MtlxInput& input : node->inputs) previous.insert(input.name, input);

    node->type = type;
    node->inputs = defaultInputsForCategory(node->category, type);
    for (MtlxInput& input : node->inputs) {
        const auto it = previous.constFind(input.name);
        if (it == previous.constEnd()) continue;
        // Keep wired connections when the port still exists; keep values when types match.
        if (!it->nodename.isEmpty()) input.nodename = it->nodename;
        if (it->type == input.type && !it->value.isEmpty()) input.value = it->value;
        else if (it->type == input.type) input.value = it->value;
    }

    preservedSelection_ = nodeName;
    writeModel(true);
    rebuild();
    emit statusMessage(QString("%1 type → %2").arg(nodeName, type));
    return true;
}

// ---------------------------------------------------------------------------
// Root-level material containers
// ---------------------------------------------------------------------------

namespace {

class MaterialContainerItem : public QGraphicsItem {
public:
    enum { Type = UserType + 72 };

    explicit MaterialContainerItem(Node* node) : node_(node) {
        setFlag(ItemIsSelectable, true);
        setFlag(ItemIsMovable, true);
        setZValue(2.0);
    }

    int type() const override { return Type; }
    Node* materialNode() const { return node_; }

    QRectF boundingRect() const override {
        return bodyRect().adjusted(-10.0, -10.0, 10.0, 42.0);
    }

    QPainterPath shape() const override {
        QPainterPath path;
        path.addRoundedRect(bodyRect(), 7.0, 7.0);
        return path;
    }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) override {
        painter->setRenderHint(QPainter::Antialiasing, true);
        const QRectF body = bodyRect();
        painter->setPen(Qt::NoPen);
        painter->setBrush(QColor(0, 0, 0, 78));
        painter->drawRoundedRect(body.translated(2.0, 3.0), 7.0, 7.0);

        const QColor bodyColor = nodeBodyColor(NodeIconKind::Material, theme::panel());
        QLinearGradient gradient(body.topLeft(), body.bottomLeft());
        gradient.setColorAt(0.0, bodyColor.lighter(112));
        gradient.setColorAt(1.0, bodyColor.darker(118));
        painter->setBrush(gradient);
        painter->setPen(QPen(isSelected() ? theme::selection() : QColor(20, 21, 24), isSelected() ? 2.0 : 1.0));
        painter->drawRoundedRect(body, 7.0, 7.0);

        painter->save();
        QPainterPath clip;
        clip.addRoundedRect(body, 7.0, 7.0);
        painter->setClipPath(clip);
        paintNodeIcon(*painter, NodeIconKind::Material,
                      QRectF(body.left() + 8.0, body.top() + 4.0, body.width() - 16.0, body.height() - 8.0));
        painter->restore();

        // Name / hint live under the tile, matching the LOP network editor.
        QFont nameFont = painter->font();
        nameFont.setPointSizeF(8.4);
        nameFont.setBold(true);
        painter->setFont(nameFont);
        painter->setPen(QColor(245, 246, 248));
        const QString name = node_ ? node_->name() : QString("material");
        const QRectF nameRect(body.left(), body.bottom() + 4.0, body.width(), 16.0);
        painter->drawText(nameRect, Qt::AlignLeft | Qt::AlignVCenter,
                          QFontMetrics(nameFont).elidedText(name, Qt::ElideRight, int(nameRect.width())));

        QFont small = painter->font();
        small.setBold(false);
        small.setPointSizeF(7.0);
        painter->setFont(small);
        painter->setPen(theme::textDim());
        painter->drawText(QRectF(body.left(), body.bottom() + 20.0, body.width(), 14.0),
                          Qt::AlignLeft | Qt::AlignVCenter, "MaterialX  ·  double-click");
    }

private:
    QRectF bodyRect() const { return QRectF(-60.0, -30.0, 120.0, 60.0); }
    Node* node_ = nullptr;
};

MaterialContainerItem* containerItemAt(QGraphicsView* view, const QPoint& viewPosition) {
    for (QGraphicsItem* item : view->items(viewPosition)) {
        if (auto* container = qgraphicsitem_cast<MaterialContainerItem*>(item)) return container;
    }
    return nullptr;
}

}  // namespace

MaterialContainerGraphView::MaterialContainerGraphView(QWidget* parent) : QGraphicsView(parent) {
    graphScene_ = new QGraphicsScene(this);
    // Large scene rect (same as Scene Network) so pan works freely on both axes.
    graphScene_->setSceneRect(-8000, -8000, 16000, 16000);
    setScene(graphScene_);
    setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    setTransformationAnchor(QGraphicsView::NoAnchor);
    setResizeAnchor(QGraphicsView::NoAnchor);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setDragMode(QGraphicsView::RubberBandDrag);
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
    setBackgroundBrush(theme::gridDark());
    connect(graphScene_, &QGraphicsScene::selectionChanged, this, &MaterialContainerGraphView::selectionChanged);
}

void MaterialContainerGraphView::setMaterials(const QVector<Node*>& materials) {
    Node* keep = selectedMaterial();
    {
        const QSignalBlocker blocker(graphScene_);
        graphScene_->clear();
        const int count = int(materials.size());
        const int columns = std::max(1, int(std::ceil(std::sqrt(double(std::max(1, count))))));
        const int rows = std::max(1, int(std::ceil(double(count) / double(columns))));
        // Center the grid on the scene origin so a default view looks framed.
        const double originX = -0.5 * double(columns - 1) * 190.0;
        const double originY = -0.5 * double(rows - 1) * 110.0;
        for (int i = 0; i < count; ++i) {
            Node* material = materials[i];
            if (!material) continue;
            auto* item = new MaterialContainerItem(material);
            const int col = i % columns;
            const int row = i / columns;
            item->setPos(originX + col * 190.0, originY + row * 110.0);
            graphScene_->addItem(item);
            if (material == keep) item->setSelected(true);
        }
        // clear() can shrink the scene rect — keep pan room on both axes.
        graphScene_->setSceneRect(-8000, -8000, 16000, 16000);
    }
    scheduleFrameGraph();
    // Only notify when the selected container identity actually changed.
    if (selectedMaterial() != keep) emit selectionChanged();
}

Node* MaterialContainerGraphView::selectedMaterial() const {
    Node* selected = nullptr;
    for (QGraphicsItem* item : graphScene_->selectedItems()) {
        if (auto* container = qgraphicsitem_cast<MaterialContainerItem*>(item)) {
            if (selected && selected != container->materialNode()) return nullptr;
            selected = container->materialNode();
        }
    }
    return selected;
}

void MaterialContainerGraphView::showEvent(QShowEvent* event) {
    QGraphicsView::showEvent(event);
    // Always reframe when the Materials tab becomes visible — tabify often left
    // containers stuck on the left after a zero-size first frame.
    scheduleFrameGraph();
}

void MaterialContainerGraphView::resizeEvent(QResizeEvent* event) {
    QGraphicsView::resizeEvent(event);
    scheduleFrameGraph();
}

void MaterialContainerGraphView::scheduleFrameGraph() {
    pendingFrame_ = true;
    auto attemptFrame = [this]() {
        if (!pendingFrame_) return;
        if (!isVisible() || width() < 80 || height() < 60) return;
        frameGraph();
    };
    QTimer::singleShot(0, this, attemptFrame);
    QTimer::singleShot(33, this, attemptFrame);
    QTimer::singleShot(100, this, attemptFrame);
}

void MaterialContainerGraphView::frameGraph() {
    if (width() < 80 || height() < 60) {
        pendingFrame_ = true;
        return;
    }
    pendingFrame_ = false;
    graphScene_->setSceneRect(-8000, -8000, 16000, 16000);
    if (graphScene_->items().isEmpty()) {
        resetTransform();
        centerOn(0.0, 0.0);
        return;
    }
    QRectF bounds;
    for (QGraphicsItem* item : graphScene_->items()) {
        if (item->type() == MaterialContainerItem::Type) bounds |= item->sceneBoundingRect();
    }
    if (bounds.isEmpty()) bounds = graphScene_->itemsBoundingRect();
    bounds = bounds.adjusted(-60.0, -60.0, 60.0, 60.0);

    const QGraphicsView::ViewportAnchor savedAnchor = transformationAnchor();
    setTransformationAnchor(QGraphicsView::AnchorViewCenter);
    resetTransform();
    fitInView(bounds, Qt::KeepAspectRatio);
    const double scaleValue = transform().m11();
    if (scaleValue > 1.35 || scaleValue < 0.5) {
        resetTransform();
        const double clamped = std::clamp(scaleValue, 0.5, 1.35);
        scale(clamped, clamped);
    }
    centerOn(bounds.center());
    setTransformationAnchor(savedAnchor);
}

void MaterialContainerGraphView::wheelEvent(QWheelEvent* event) {
    const qreal factor = graphicsViewWheelZoomFactor(event);
    zoomGraphicsViewAtCursor(this, factor, event->globalPosition(),
                             panning_ ? &panScenePoint_ : nullptr, 0.16, 4.0);
    event->accept();
}

void MaterialContainerGraphView::keyPressEvent(QKeyEvent* event) {
    switch (event->key()) {
        case Qt::Key_Space:
            if (!event->isAutoRepeat()) {
                spacePressed_ = true;
                if (!panning_) viewport()->setCursor(Qt::OpenHandCursor);
            }
            event->accept();
            return;
        case Qt::Key_F:
            frameGraph();
            event->accept();
            return;
        default:
            break;
    }
    QGraphicsView::keyPressEvent(event);
}

void MaterialContainerGraphView::keyReleaseEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Space) {
        spacePressed_ = false;
        if (panning_) endPan();
        event->accept();
        return;
    }
    QGraphicsView::keyReleaseEvent(event);
}

bool MaterialContainerGraphView::shouldBeginPan(const QMouseEvent* event) const {
    // Match Scene Network (Houdini-style):
    //   MMB drag       — pan
    //   Alt+LMB drag   — pan
    //   Space+LMB drag — pan
    if (event->button() == Qt::MiddleButton) return true;
    if (event->button() == Qt::LeftButton && (event->modifiers() & Qt::AltModifier)) return true;
    if (event->button() == Qt::LeftButton && spacePressed_) return true;
    return false;
}

void MaterialContainerGraphView::beginPan(const QPointF& globalPos) {
    panning_ = true;
    savedDragMode_ = dragMode();
    savedAnchor_ = transformationAnchor();
    setDragMode(QGraphicsView::NoDrag);
    setTransformationAnchor(QGraphicsView::NoAnchor);
    setResizeAnchor(QGraphicsView::NoAnchor);
    viewport()->grabMouse();
    viewport()->setCursor(Qt::ClosedHandCursor);
    setFocus(Qt::MouseFocusReason);
    panScenePoint_ = mapToScene(graphicsViewViewportPos(this, globalPos));
}

void MaterialContainerGraphView::updatePan(const QPointF& globalPos) {
    if (!panning_) return;
    glueGraphicsViewPan(this, panScenePoint_, globalPos);
}

void MaterialContainerGraphView::endPan() {
    if (!panning_) return;
    panning_ = false;
    if (QWidget::mouseGrabber() == viewport()) viewport()->releaseMouse();
    setDragMode(savedDragMode_);
    setTransformationAnchor(savedAnchor_);
    viewport()->unsetCursor();
}

void MaterialContainerGraphView::mousePressEvent(QMouseEvent* event) {
    if (shouldBeginPan(event)) {
        beginPan(event->globalPosition());
        event->accept();
        return;
    }
    QGraphicsView::mousePressEvent(event);
}

void MaterialContainerGraphView::mouseMoveEvent(QMouseEvent* event) {
    if (panning_) {
        updatePan(event->globalPosition());
        event->accept();
        return;
    }
    QGraphicsView::mouseMoveEvent(event);
}

void MaterialContainerGraphView::mouseReleaseEvent(QMouseEvent* event) {
    if (panning_) {
        endPan();
        event->accept();
        return;
    }
    QGraphicsView::mouseReleaseEvent(event);
}

void MaterialContainerGraphView::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        if (MaterialContainerItem* item = containerItemAt(this, event->pos())) {
            graphScene_->clearSelection();
            item->setSelected(true);
            emit diveRequested(item->materialNode());
            event->accept();
            return;
        }
    }
    QGraphicsView::mouseDoubleClickEvent(event);
}

void MaterialContainerGraphView::drawBackground(QPainter* painter, const QRectF& rect) {
    painter->fillRect(rect, theme::gridDark());
    const qreal step = 40.0;
    painter->setPen(QPen(theme::gridLine(), 0.0));
    const qreal left = std::floor(rect.left() / step) * step;
    const qreal top = std::floor(rect.top() / step) * step;
    for (qreal x = left; x < rect.right(); x += step)
        painter->drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()));
    for (qreal y = top; y < rect.bottom(); y += step)
        painter->drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y));

    if (graphScene_->items().isEmpty()) {
        painter->resetTransform();
        QFont font = painter->font();
        font.setPointSizeF(11.0);
        painter->setFont(font);
        painter->setPen(theme::textDim());
        painter->drawText(QRect(0, 0, width(), height()), Qt::AlignCenter,
                          "No material nodes in the scene\nAdd a Material node in the Scene Network");
    }
}

// ---------------------------------------------------------------------------
// Public dock widget: containers + MaterialX dive canvas
// ---------------------------------------------------------------------------

MaterialNetworkView::MaterialNetworkView(QWidget* parent) : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    auto* chrome = new QWidget(this);
    chrome->setObjectName("materialNetworkChrome");
    auto* chromeLayout = new QHBoxLayout(chrome);
    chromeLayout->setContentsMargins(6, 4, 8, 4);
    chromeLayout->setSpacing(6);

    backButton_ = new QToolButton(chrome);
    backButton_->setText(QString::fromUtf8("↑"));
    backButton_->setToolTip("Up to material containers");
    backButton_->setFixedSize(28, 24);
    backButton_->setEnabled(false);
    connect(backButton_, &QToolButton::clicked, this, &MaterialNetworkView::goUp);
    chromeLayout->addWidget(backButton_);

    presetsButton_ = new QToolButton(chrome);
    presetsButton_->setText("Presets");
    presetsButton_->setToolTip("Apply material presets (metals / glass) to the selected "
                               "standard_surface node");
    presetsButton_->setEnabled(false);
    connect(presetsButton_, &QToolButton::clicked, this, &MaterialNetworkView::showPresetsMenu);
    chromeLayout->addWidget(presetsButton_);

    pathLabel_ = new QLabel("Materials", chrome);
    pathLabel_->setStyleSheet("color: #c8ccd2; font-weight: 600;");
    chromeLayout->addWidget(pathLabel_, 1);
    root->addWidget(chrome);

    containerView_ = new MaterialContainerGraphView(this);
    graphView_ = new MaterialNetworkGraphView(this);
    graphView_->hide();
    root->addWidget(containerView_, 1);
    root->addWidget(graphView_, 1);

    connect(containerView_, &MaterialContainerGraphView::selectionChanged, this,
            &MaterialNetworkView::onContainerSelectionChanged);
    connect(containerView_, &MaterialContainerGraphView::diveRequested, this, &MaterialNetworkView::diveInto);
    connect(containerView_, &MaterialContainerGraphView::statusMessage, this, &MaterialNetworkView::statusMessage);
    connect(graphView_, &MaterialNetworkGraphView::materialEdited, this, &MaterialNetworkView::materialEdited);
    connect(graphView_, &MaterialNetworkGraphView::materialLayoutChanged, this, [this] {
        if (graph_) graph_->setModified(true);
    });
    connect(graphView_, &MaterialNetworkGraphView::statusMessage, this, &MaterialNetworkView::statusMessage);
    connect(graphView_, &MaterialNetworkGraphView::selectionChanged, this,
            &MaterialNetworkView::onMaterialXSelectionChanged);
    connect(graphView_, &MaterialNetworkGraphView::upRequested, this, &MaterialNetworkView::goUp);

    updateChrome();
}

void MaterialNetworkView::setGraph(NodeGraph* graph) {
    if (nodeAddedConnection_) disconnect(nodeAddedConnection_);
    if (nodeRemovedConnection_) disconnect(nodeRemovedConnection_);
    if (connectionsChangedConnection_) disconnect(connectionsChangedConnection_);
    graph_ = graph;
    if (graph_) {
        // Listen only to topology — not every parameter edit (graphChanged),
        // otherwise container refresh steals the Parameters selection.
        nodeAddedConnection_ =
            connect(graph_, &NodeGraph::nodeAdded, this, [this](Node*) { onGraphTopologyChanged(); });
        nodeRemovedConnection_ = connect(graph_, &NodeGraph::nodeAboutToBeRemoved, this, [this](Node* node) {
            if (node == currentMaterial_) goUp();
            onGraphTopologyChanged();
        });
        connectionsChangedConnection_ =
            connect(graph_, &NodeGraph::connectionsChanged, this, &MaterialNetworkView::onGraphTopologyChanged);
    }
    currentMaterial_ = nullptr;
    graphView_->setMaterialNode(nullptr);
    graphView_->hide();
    containerView_->show();
    refreshContainers();
    updateChrome();
    emit selectionChanged();
}

void MaterialNetworkView::refreshContainers() {
    QVector<Node*> materials;
    if (graph_) {
        for (const NodePtr& node : graph_->nodes()) {
            if (node && node->typeName() == "material") materials.push_back(node.get());
        }
    }
    std::sort(materials.begin(), materials.end(),
              [](Node* a, Node* b) { return a->name().localeAwareCompare(b->name()) < 0; });
    containerView_->setMaterials(materials);
}

void MaterialNetworkView::diveInto(Node* material) {
    if (!material || material->typeName() != "material") return;
    currentMaterial_ = material;
    containerView_->hide();
    graphView_->show();
    graphView_->setMaterialNode(material);
    // Force center after dive — rebuild may have framed while the view was still tiny.
    graphView_->requestFrame();
    updateChrome();
    emit selectionChanged();
}

void MaterialNetworkView::goUp() {
    if (!currentMaterial_) return;
    currentMaterial_ = nullptr;
    graphView_->setMaterialNode(nullptr);
    graphView_->hide();
    containerView_->show();
    refreshContainers();
    updateChrome();
    emit selectionChanged();
}

void MaterialNetworkView::updateChrome() {
    const bool inside = currentMaterial_ != nullptr;
    backButton_->setEnabled(inside);
    bool canPreset = false;
    if (inside && graphView_) {
        const MaterialNetworkGraphView::MtlxNode* n = graphView_->selectedNode();
        canPreset = n && (n->category == "standard_surface" || n->category == "dielectric_bsdf" ||
                          n->category == "conductor_bsdf");
    }
    if (presetsButton_) presetsButton_->setEnabled(canPreset);
    if (inside)
        pathLabel_->setText(QString("Materials  /  %1").arg(currentMaterial_->name()));
    else
        pathLabel_->setText("Materials");
}

void MaterialNetworkView::onContainerSelectionChanged() { emit selectionChanged(); }

void MaterialNetworkView::onMaterialXSelectionChanged() {
    updateChrome();
    emit selectionChanged();
}

void MaterialNetworkView::showPresetsMenu() {
    if (!presetsButton_ || !presetsButton_->isEnabled()) return;
    QMenu menu(this);
    auto* metals = menu.addMenu("Metals (conductor η/κ)");
    metals->addAction("Gold (Au)", this, [this] { applyPreset("Au"); });
    metals->addAction("Silver (Ag)", this, [this] { applyPreset("Ag"); });
    metals->addAction("Copper (Cu)", this, [this] { applyPreset("Cu"); });
    metals->addAction("Aluminium (Al)", this, [this] { applyPreset("Al"); });
    auto* glass = menu.addMenu("Glass / Dielectric");
    glass->addAction("Clear Glass (IOR 1.5, Abbe 55)", this, [this] { applyPreset("glass_clear"); });
    glass->addAction("Crown Glass (IOR 1.52, Abbe 60)", this, [this] { applyPreset("glass_crown"); });
    glass->addAction("Flint Glass (IOR 1.65, Abbe 33)", this, [this] { applyPreset("glass_flint"); });
    glass->addAction("Acrylic (IOR 1.49, Abbe 55)", this, [this] { applyPreset("glass_acrylic"); });
    menu.exec(presetsButton_->mapToGlobal(QPoint(0, presetsButton_->height())));
}

void MaterialNetworkView::applyPreset(const QString& presetId) {
    if (!graphView_ || !currentMaterial_) return;
    const MaterialNetworkGraphView::MtlxNode* node = graphView_->selectedNode();
    if (!node) return;
    const QString name = node->name;

    auto setIn = [&](const QString& input, const QString& value) {
        graphView_->setInputValue(name, input, value);
    };
    auto setNk = [&](const char* metal) {
        Vec3 eta, k;
        metalNkRgbPreset(metal, eta, k);
        setIn("conductor_eta", QString("%1, %2, %3").arg(eta.x).arg(eta.y).arg(eta.z));
        setIn("conductor_k", QString("%1, %2, %3").arg(k.x).arg(k.y).arg(k.z));
    };

    if (presetId == "Au") {
        setIn("base_color", "1, 0.71, 0.29");
        setIn("metalness", "1");
        setIn("specular_roughness", "0.2");
        setIn("transmission", "0");
        setNk("Au");
    } else if (presetId == "Ag") {
        setIn("base_color", "0.97, 0.96, 0.91");
        setIn("metalness", "1");
        setIn("specular_roughness", "0.15");
        setIn("transmission", "0");
        setNk("Ag");
    } else if (presetId == "Cu") {
        setIn("base_color", "0.95, 0.64, 0.54");
        setIn("metalness", "1");
        setIn("specular_roughness", "0.25");
        setIn("transmission", "0");
        setNk("Cu");
    } else if (presetId == "Al") {
        setIn("base_color", "0.91, 0.92, 0.92");
        setIn("metalness", "1");
        setIn("specular_roughness", "0.2");
        setIn("transmission", "0");
        setNk("Al");
    } else if (presetId == "glass_clear") {
        setIn("base_color", "1, 1, 1");
        setIn("metalness", "0");
        setIn("specular_roughness", "0.0");
        setIn("transmission", "1");
        setIn("specular_IOR", "1.5");
        setIn("dispersion_abbe", "55");
        setIn("conductor_eta", "1.5, 1.5, 1.5");
        setIn("conductor_k", "0, 0, 0");
    } else if (presetId == "glass_crown") {
        setIn("base_color", "1, 1, 1");
        setIn("metalness", "0");
        setIn("specular_roughness", "0.0");
        setIn("transmission", "1");
        setIn("specular_IOR", "1.52");
        setIn("dispersion_abbe", "60");
        setIn("conductor_eta", "1.5, 1.5, 1.5");
        setIn("conductor_k", "0, 0, 0");
    } else if (presetId == "glass_flint") {
        setIn("base_color", "1, 1, 1");
        setIn("metalness", "0");
        setIn("specular_roughness", "0.0");
        setIn("transmission", "1");
        setIn("specular_IOR", "1.65");
        setIn("dispersion_abbe", "33");
        setIn("conductor_eta", "1.5, 1.5, 1.5");
        setIn("conductor_k", "0, 0, 0");
    } else if (presetId == "glass_acrylic") {
        setIn("base_color", "1, 1, 1");
        setIn("metalness", "0");
        setIn("specular_roughness", "0.0");
        setIn("transmission", "1");
        setIn("specular_IOR", "1.49");
        setIn("dispersion_abbe", "55");
        setIn("conductor_eta", "1.5, 1.5, 1.5");
        setIn("conductor_k", "0, 0, 0");
    }

    // Clear any legacy spectral metal tag.
    if (currentMaterial_->findParameter("spectralmetalpreset"))
        currentMaterial_->setParameterValue("spectralmetalpreset", 0);
    emit materialEdited(currentMaterial_);
    emit statusMessage(QString("Applied preset: %1").arg(presetId));
}

void MaterialNetworkView::onGraphTopologyChanged() {
    if (currentMaterial_ && graph_ && !graph_->findNode(currentMaterial_->name())) {
        goUp();
        return;
    }
    if (!currentMaterial_) refreshContainers();
    else updateChrome();
}

Node* MaterialNetworkView::selectedLopNode() const {
    if (currentMaterial_) return currentMaterial_;
    return containerView_ ? containerView_->selectedMaterial() : nullptr;
}

bool MaterialNetworkView::selectedMaterialX(MaterialXSelection& out) const {
    out = {};
    if (!currentMaterial_ || !graphView_) return false;
    const MaterialNetworkGraphView::MtlxNode* node = graphView_->selectedNode();
    if (!node) return false;
    out.hostMaterial = currentMaterial_;
    out.category = node->category;
    out.type = node->type;
    out.name = node->name;
    out.activeIntegrator = 0;
    if (graph_) {
        for (const NodePtr& n : graph_->nodes()) {
            if (n && n->typeName() == QLatin1String("rendersettings")) {
                out.activeIntegrator = n->intValue("integrator", 0);
                break;
            }
        }
    }
    if (const MaterialXNodeCatalogEntry* entry = findCatalogEntry(node->category)) {
        out.typeVariants = entry->typeVariants;
        if (out.typeVariants.isEmpty() && !entry->type.isEmpty()) out.typeVariants << entry->type;
    } else if (!node->type.isEmpty()) {
        out.typeVariants << node->type;
    }
    out.inputs.reserve(node->inputs.size());
    bool triplanarPerAxis = false;
    if (node->category == "triplanarprojection") {
        for (const MaterialNetworkGraphView::MtlxInput& input : node->inputs) {
            if (input.name == "input_per_axis") {
                triplanarPerAxis = (input.value == "true" || input.value == "1");
                break;
            }
        }
    }
    for (const MaterialNetworkGraphView::MtlxInput& input : node->inputs) {
        // Arnold: hide per-axis file slots until Input Per Axis is enabled.
        if (node->category == "triplanarprojection" && !triplanarPerAxis &&
            (input.name == "filex" || input.name == "filey" || input.name == "filez")) {
            continue;
        }
        MaterialXInputParam param;
        param.name = input.name;
        param.type = input.type;
        param.value = input.value;
        param.nodename = input.nodename;
        out.inputs.push_back(param);
    }
    return true;
}

bool MaterialNetworkView::renameSelectedMaterialX(const QString& newName) {
    if (!graphView_) return false;
    const QString oldName = graphView_->selectedNodeName();
    if (oldName.isEmpty()) return false;
    return graphView_->renameNode(oldName, newName);
}

bool MaterialNetworkView::setSelectedMaterialXInput(const QString& inputName, const QString& value) {
    if (!graphView_) return false;
    const QString nodeName = graphView_->selectedNodeName();
    if (nodeName.isEmpty()) return false;
    return graphView_->setInputValue(nodeName, inputName, value);
}

bool MaterialNetworkView::setSelectedMaterialXType(const QString& type) {
    if (!graphView_) return false;
    const QString nodeName = graphView_->selectedNodeName();
    if (nodeName.isEmpty()) return false;
    return graphView_->setNodeType(nodeName, type);
}

}  // namespace sol

#include "material_network_view.moc"
