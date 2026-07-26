#include "ui/material_network_view.h"

#include <QContextMenuEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QLineF>
#include <QMap>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QRegularExpression>
#include <QResizeEvent>
#include <QScrollBar>
#include <QShowEvent>
#include <QSignalBlocker>
#include <QStyleOptionGraphicsItem>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>
#include <limits>

#include "io/image_io.h"
#include "io/materialx_graph.h"
#include "nodes/node.h"
#include "nodes/node_graph.h"
#include "nodes/parameter.h"
#include "ui/material_wire_item.h"
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

const MaterialXNodeCatalogEntry* findCatalogEntry(const QString& category, const QString& type = QString()) {
    const QVector<MaterialXNodeCatalogEntry>& catalog = catalogCache();
    const MaterialXNodeCatalogEntry* fallback = nullptr;
    for (const MaterialXNodeCatalogEntry& entry : catalog) {
        if (entry.category != category) continue;
        if (!type.isEmpty() && entry.type == type) return &entry;
        if (!fallback) fallback = &entry;
        // Prefer color3 / surfaceshader / material / float variants when type omitted.
        if (type.isEmpty()) {
            if (entry.type == "color3" || entry.type == "surfaceshader" || entry.type == "material")
                return &entry;
        }
    }
    return fallback;
}

bool isKnownMaterialXCategory(const QString& category) {
    if (category.isEmpty() || category == "materialx" || category == "nodegraph" || category == "nodedef" ||
        category == "implementation" || category == "backdrop")
        return false;
    if (findCatalogEntry(category)) return true;
    // Keep previously hardcoded essentials even if libraries failed to load.
    return category == "standard_surface" || category == "surfacematerial" || category == "image" ||
           category == "constant" || category == "multiply" || category == "mix" || category == "normalmap" ||
           category == "tiledimage" || category == "add" || category == "texcoord";
}

QColor colorForCategory(const QString& category) {
    if (category == "image" || category == "tiledimage") return QColor(42, 132, 132);
    if (category == "standard_surface") return QColor(189, 116, 45);
    if (category == "surfacematerial") return QColor(126, 82, 170);
    if (category == "normalmap") return QColor(96, 101, 108);
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
    return type != "filename";
}

QString fallbackDefaultDocument() {
    return QStringLiteral(
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.38\">\n"
        "  <standard_surface name=\"standard_surface1\" type=\"surfaceshader\" xpos=\"0\" ypos=\"0\">\n"
        "    <input name=\"base_color\" type=\"color3\" value=\"0.8, 0.8, 0.8\"/>\n"
        "    <input name=\"specular_roughness\" type=\"float\" value=\"0.35\"/>\n"
        "    <input name=\"metalness\" type=\"float\" value=\"0\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"surface\" type=\"material\" xpos=\"4\" ypos=\"0\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"standard_surface1\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n");
}

QPointF defaultLayoutForCategory(const QString& category, int ordinal) {
    if (category == "surfacematerial") return QPointF(4.0, 0.0);
    if (category == "standard_surface") return QPointF(0.0, 0.0);
    if (category == "image") return QPointF(-4.0, qreal(ordinal) * 1.2);
    if (category == "normalmap") return QPointF(-2.0, qreal(ordinal) * 1.2);
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
        int modelIndex = -1;
        bool connected = false;
    };

    MaterialNetworkNodeItem(QString nodeName, QString category, QString typeName, QVector<InputPort> inputs,
                            QString subtitle)
        : nodeName_(std::move(nodeName)),
          category_(std::move(category)),
          typeName_(std::move(typeName)),
          inputs_(std::move(inputs)),
          subtitle_(std::move(subtitle)) {
        setFlag(ItemIsSelectable, true);
        setFlag(ItemIsMovable, true);
        setFlag(ItemSendsGeometryChanges, true);
        setCursor(Qt::OpenHandCursor);
        setZValue(2.0);
    }

    int type() const override { return Type; }
    const QString& nodeName() const { return nodeName_; }
    const QString& category() const { return category_; }

    QRectF boundingRect() const override { return bodyRect().adjusted(-16.0, -14.0, 16.0, 14.0); }

    QPainterPath shape() const override {
        QPainterPath path;
        path.addRoundedRect(bodyRect(), 6.0, 6.0);
        for (int i = 0; i < inputs_.size(); ++i) path.addEllipse(inputPortLocal(i), kPortHitRadius, kPortHitRadius);
        path.addEllipse(outputPortLocal(), kPortHitRadius, kPortHitRadius);
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
        painter->restore();

        QFont nameFont = painter->font();
        nameFont.setPointSizeF(8.2);
        nameFont.setBold(true);
        painter->setFont(nameFont);
        painter->setPen(theme::text());
        const QRectF nameRect(body.left() + 8.0, body.top() + 2.0, body.width() - 16.0, 16.0);
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
        auto drawPort = [painter](QPointF local, bool connected, QColor color) {
            painter->setPen(Qt::NoPen);
            painter->setBrush(QColor(color.red(), color.green(), color.blue(), connected ? 76 : 42));
            painter->drawEllipse(local, kPortRadius + 3.0, kPortRadius + 3.0);
            painter->setPen(QPen(QColor(17, 18, 22), 1.0));
            painter->setBrush(connected ? theme::wireActive() : QColor(128, 132, 140));
            painter->drawEllipse(local, kPortRadius, kPortRadius);
        };

        const QColor categoryColor = colorForCategory(category_);
        for (int i = 0; i < inputs_.size(); ++i) drawPort(inputPortLocal(i), inputs_[i].connected, categoryColor);
        drawPort(outputPortLocal(), true, categoryColor);
    }

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

void addWire(QGraphicsScene* scene, QPointF from, QPointF to, const QString& targetName, const QString& inputName) {
    auto* wire = new MaterialWireItem(targetName, inputName);
    wire->setWirePath(makeWirePath(from, to));
    scene->addItem(wire);
}

}  // namespace

MaterialNetworkGraphView::MaterialNetworkGraphView(QWidget* parent) : QGraphicsView(parent) {
    graphScene_ = new QGraphicsScene(this);
    // Large scene rect (same idea as Network Editor) so pan works freely on both axes.
    graphScene_->setSceneRect(-8000, -8000, 16000, 16000);
    setScene(graphScene_);
    setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);
    setViewportUpdateMode(QGraphicsView::SmartViewportUpdate);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setDragMode(QGraphicsView::RubberBandDrag);
    setFocusPolicy(Qt::StrongFocus);
    setMinimumHeight(180);
    lastMousePoint_ = viewport()->rect().center();
    selectionConnection_ = connect(graphScene_, &QGraphicsScene::selectionChanged, this,
                                   &MaterialNetworkGraphView::emitSelectionChanged);
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
    materialNode_->setParameterValue("mtlx", xml);
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

QString MaterialNetworkGraphView::defaultTypeForCategory(const QString& category) {
    if (const MaterialXNodeCatalogEntry* entry = findCatalogEntry(category)) return entry->type;
    if (category == "standard_surface") return "surfaceshader";
    if (category == "surfacematerial") return "material";
    if (category == "normalmap") return "vector3";
    if (category == "texcoord") return "vector2";
    return "color3";
}

QVector<MaterialNetworkGraphView::MtlxInput> MaterialNetworkGraphView::defaultInputsForCategory(const QString& category,
                                                                                      const QString& type) {
    QVector<MtlxInput> inputs;
    if (const MaterialXNodeCatalogEntry* entry = findCatalogEntry(category, type)) {
        for (const MaterialXNodeInputDef& def : entry->inputs) {
            // Keep standard_surface UI focused on the common shading ports; full nodedef is huge.
            if (category == "standard_surface") {
                static const QStringList keep = {"base_color", "specular_roughness", "metalness", "specular",
                                                 "specular_IOR", "transmission", "opacity", "emission",
                                                 "emission_color", "normal", "subsurface", "subsurface_color",
                                                 "subsurface_radius"};
                if (!keep.contains(def.name)) continue;
            }
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
    } else if (category == "standard_surface") {
        inputs.push_back({"base_color", "color3", "0.8, 0.8, 0.8", {}});
        inputs.push_back({"specular_roughness", "float", "0.35", {}});
        inputs.push_back({"metalness", "float", "0", {}});
        inputs.push_back({"normal", "vector3", {}, {}});
    } else if (category == "surfacematerial") {
        inputs.push_back({"surfaceshader", "surfaceshader", {}, {}});
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
            for (const MtlxInput& input : defaultInputsForCategory(node.category, node.type))
                ensureInput(node.inputs, input.name, input.type, input.value);
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
    ensureInput(surface->inputs, "surfaceshader", "surfaceshader");
    MtlxInput* surfaceShaderInput = nullptr;
    for (MtlxInput& input : surface->inputs) {
        if (input.name == "surfaceshader") {
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
            ports.push_back({input.name, i, !input.nodename.isEmpty()});
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

        auto* item = new MaterialNetworkNodeItem(node.name, node.category, node.type, ports, subtitle);
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
                            target.name, input.name);
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
    pendingFrame_ = true;
    if (isVisible() && width() > 50) frameGraph();
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
    if (pendingFrame_ && width() > 50) frameGraph();
}

void MaterialNetworkGraphView::resizeEvent(QResizeEvent* event) {
    QGraphicsView::resizeEvent(event);
    if (pendingFrame_ && width() > 50) frameGraph();
}

void MaterialNetworkGraphView::frameGraph() {
    pendingFrame_ = false;
    const QRectF bounds = graphScene_->itemsBoundingRect();
    resetTransform();
    if (bounds.isEmpty()) {
        centerOn(0.0, 0.0);
        return;
    }
    fitInView(bounds.adjusted(-80.0, -80.0, 80.0, 80.0), Qt::KeepAspectRatio);
    const double scaleValue = transform().m11();
    if (scaleValue > 1.0 || scaleValue < 0.55) {
        resetTransform();
        const double clamped = std::clamp(scaleValue, 0.55, 1.0);
        scale(clamped, clamped);
        centerOn(bounds.center());
    }
}

void MaterialNetworkGraphView::wheelEvent(QWheelEvent* event) {
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    const QPoint delta = event->angleDelta().y() != 0 ? event->angleDelta() : event->pixelDelta();
    const qreal steps = qreal(delta.y()) / 120.0;
    if (std::abs(steps) < 1e-4) {
        event->accept();
        return;
    }

    const qreal factor = std::pow(1.08, steps);
    const double newScale = transform().m11() * factor;
    if (newScale < 0.16 || newScale > 4.0) {
        event->accept();
        return;
    }
    QGraphicsView::scale(factor, factor);
    event->accept();
}

void MaterialNetworkGraphView::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Space && !event->isAutoRepeat()) {
        spacePressed_ = true;
        if (!panning_) viewport()->setCursor(Qt::OpenHandCursor);
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_Tab) {
        const QPoint position = viewport()->rect().contains(lastMousePoint_) ? lastMousePoint_ : viewport()->rect().center();
        showAddNodeMenu(position);
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_Delete || event->key() == Qt::Key_Backspace) {
        deleteSelectedNodes();
        event->accept();
        return;
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
        emit statusMessage("Select a Material node in the Network Editor");
        return;
    }

    QMenu menu(this);
    QMap<QString, QMenu*> groupMenus;
    const QVector<MaterialXNodeCatalogEntry>& catalog = catalogCache();
    if (catalog.isEmpty()) {
        emit statusMessage("MaterialX node catalog is empty");
        return;
    }

    for (const MaterialXNodeCatalogEntry& entry : catalog) {
        QMenu* groupMenu = groupMenus.value(entry.group);
        if (!groupMenu) {
            groupMenu = menu.addMenu(entry.group);
            groupMenus.insert(entry.group, groupMenu);
        }
        QAction* action = groupMenu->addAction(entry.label);
        action->setData(QStringList{entry.category, entry.type});
    }

    QAction* chosen = menu.exec(viewport()->mapToGlobal(viewPosition));
    if (!chosen) return;
    const QStringList data = chosen->data().toStringList();
    if (data.size() < 2) return;
    addNode(data[0], data[1], mapToScene(viewPosition));
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
    if (!target || inputIndex < 0 || inputIndex >= target->inputs.size()) return;
    if (!isConnectableInput(target->inputs[inputIndex].name, target->inputs[inputIndex].type)) return;
    target->inputs[inputIndex].nodename = sourceName;
    target->inputs[inputIndex].value.clear();
    writeModel(true);
    rebuild();
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
        rebuild();
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

void MaterialNetworkGraphView::mousePressEvent(QMouseEvent* event) {
    lastMousePoint_ = event->pos();
    mousePressPoint_ = event->pos();
    mouseMovedSincePress_ = false;
    clickImageNode_.clear();

    if (shouldBeginPan(event)) {
        beginPan(event->pos());
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton) {
        if (MaterialNetworkNodeItem* source =
                outputPortAt(graphScene_, mapToScene(event->pos()), kPortHitRadius * 1.8)) {
            beginWire(source->nodeName(), source->outputPortScene());
            event->accept();
            return;
        }
        // Pull existing wire off an input (rewire), matching the Network Editor.
        if (const InputHit hit = inputPortAt(graphScene_, mapToScene(event->pos()), kPortHitRadius * 1.8);
            hit.item) {
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
                        updateWire(mapToScene(event->pos()));
                        event->accept();
                        return;
                    }
                }
            }
        }
        if (MaterialNetworkNodeItem* item = nodeItemAt(this, event->pos())) {
            if (item->category() == "image" || item->category() == "tiledimage")
                clickImageNode_ = item->nodeName();
        }
    }

    QGraphicsView::mousePressEvent(event);
}

void MaterialNetworkGraphView::mouseMoveEvent(QMouseEvent* event) {
    lastMousePoint_ = event->pos();
    if (QLineF(event->pos(), mousePressPoint_).length() > 4.0) mouseMovedSincePress_ = true;

    if (panning_) {
        updatePan(event->pos());
        event->accept();
        return;
    }
    if (wiring_) {
        updateWire(mapToScene(event->pos()));
        event->accept();
        return;
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

    const QString filter = "Images (*.png *.jpg *.jpeg *.exr *.hdr *.tif *.tiff *.bmp *.webp);;All Files (*)";
    const QString path =
        QFileDialog::getOpenFileName(this, "Choose texture for " + nodeName, node->inputs[fileInput].value, filter);
    if (path.isEmpty()) return;

    // MaterialX authoring: store unresolved <UDIM> + geominfo udimset (not one concrete tile).
    node->inputs[fileInput].value = applyUdimFilename(path);
    node->inputs[fileInput].nodename.clear();
    writeModel(true);
    rebuild();
    emit statusMessage(QString("%1 file set to %2").arg(nodeName, QFileInfo(node->inputs[fileInput].value).fileName()));
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
    writeModel(true);
    rebuild();
}

bool MaterialNetworkGraphView::shouldBeginPan(const QMouseEvent* event) const {
    // Match the Network Editor (Houdini-style):
    //   MMB drag       — pan
    //   Alt+LMB drag   — pan
    //   Space+LMB drag — pan
    if (event->button() == Qt::MiddleButton) return true;
    if (event->button() == Qt::LeftButton && (event->modifiers() & Qt::AltModifier)) return true;
    if (event->button() == Qt::LeftButton && spacePressed_) return true;
    return false;
}

void MaterialNetworkGraphView::beginPan(const QPoint& viewPosition) {
    panning_ = true;
    lastPanPoint_ = viewPosition;
    savedDragMode_ = dragMode();
    savedAnchor_ = transformationAnchor();
    setDragMode(QGraphicsView::NoDrag);
    // NoAnchor keeps 1:1 hand-drag — AnchorUnderMouse would warp the pan.
    setTransformationAnchor(QGraphicsView::NoAnchor);
    viewport()->grabMouse();
    viewport()->setCursor(Qt::ClosedHandCursor);
}

void MaterialNetworkGraphView::updatePan(const QPoint& viewPosition) {
    if (!panning_) return;
    if (viewPosition == lastPanPoint_) return;
    // Sticky hand: keep the scene point under the cursor glued to it.
    const QPointF delta = mapToScene(viewPosition) - mapToScene(lastPanPoint_);
    translate(delta.x(), delta.y());
    lastPanPoint_ = viewPosition;
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
    previewWire_ = graphScene_->addPath(makeWirePath(sourcePosition, sourcePosition),
                                        QPen(theme::wireActive(), 1.8, Qt::DashLine, Qt::RoundCap, Qt::RoundJoin));
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
        // Soft snap: drop on a node body and pick nearest free/connectable input.
        if (MaterialNetworkNodeItem* item = nodeItemAt(this, viewPosition)) {
            if (item->nodeName() != sourceName && item->inputCount() > 0) {
                int port = item->hitInputPort(scenePosition, kPortSnapRadius * 2.0);
                if (port < 0) port = 0;
                hit = {item, port};
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
    painter->resetTransform();

    QFont font = painter->font();
    if (!materialNode_) {
        font.setPointSizeF(11.0);
        painter->setFont(font);
        painter->setPen(theme::textDim());
        painter->drawText(QRect(0, 0, width(), height()), Qt::AlignCenter,
                          "Double-click a material container to edit its MaterialX graph");
        return;
    }

    font.setPointSizeF(8.0);
    painter->setFont(font);
    painter->setPen(theme::textDim());
    painter->drawText(QRect(8, height() - 22, width() - 16, 18), Qt::AlignLeft,
                      "Tab: add   MMB/Alt+LMB/Space+LMB: pan   Wheel: zoom   Del: delete   image: file");
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
        if (input.type == "filename" || inputName == "file") stored = applyUdimFilename(value);
        if (input.value == stored && !wasConnected) return true;
        input.value = stored;
        input.nodename.clear();
        writeModel(true);
        // Rebuild only when the graph appearance changes (wires / file subtitle).
        if (wasConnected || input.type == "filename") {
            preservedSelection_ = nodeName;
            rebuild();
        }
        return true;
    }
    return false;
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
        setCursor(Qt::OpenHandCursor);
        setZValue(2.0);
    }

    int type() const override { return Type; }
    Node* materialNode() const { return node_; }

    QRectF boundingRect() const override { return bodyRect().adjusted(-10.0, -10.0, 10.0, 10.0); }

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

        QLinearGradient gradient(body.topLeft(), body.bottomLeft());
        gradient.setColorAt(0.0, theme::panelLight().lighter(110));
        gradient.setColorAt(1.0, theme::panel().darker(112));
        painter->setBrush(gradient);
        painter->setPen(QPen(isSelected() ? theme::selection() : QColor(20, 21, 24), isSelected() ? 2.0 : 1.0));
        painter->drawRoundedRect(body, 7.0, 7.0);

        painter->setPen(Qt::NoPen);
        painter->setBrush(colorForCategory("material"));
        painter->drawRoundedRect(QRectF(body.left(), body.top(), body.width(), 22.0), 7.0, 7.0);
        painter->drawRect(QRectF(body.left(), body.top() + 12.0, body.width(), 10.0));

        QFont nameFont = painter->font();
        nameFont.setPointSizeF(8.4);
        nameFont.setBold(true);
        painter->setFont(nameFont);
        painter->setPen(theme::text());
        const QString name = node_ ? node_->name() : QString("material");
        painter->drawText(QRectF(body.left() + 10.0, body.top() + 3.0, body.width() - 20.0, 16.0),
                          Qt::AlignLeft | Qt::AlignVCenter,
                          QFontMetrics(nameFont).elidedText(name, Qt::ElideRight, int(body.width() - 20.0)));

        QFont small = painter->font();
        small.setBold(false);
        small.setPointSizeF(7.2);
        painter->setFont(small);
        painter->setPen(theme::textDim());
        painter->drawText(QRectF(body.left() + 10.0, body.top() + 28.0, body.width() - 20.0, 16.0),
                          Qt::AlignLeft | Qt::AlignVCenter, "material  ·  container");
        painter->drawText(QRectF(body.left() + 10.0, body.top() + 46.0, body.width() - 20.0, 16.0),
                          Qt::AlignLeft | Qt::AlignVCenter, "double-click to dive");
    }

private:
    QRectF bodyRect() const { return QRectF(-78.0, -36.0, 156.0, 72.0); }
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
    setScene(graphScene_);
    setRenderHint(QPainter::Antialiasing, true);
    setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    setDragMode(QGraphicsView::RubberBandDrag);
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
        for (int i = 0; i < count; ++i) {
            Node* material = materials[i];
            if (!material) continue;
            auto* item = new MaterialContainerItem(material);
            const int col = i % columns;
            const int row = i / columns;
            item->setPos(col * 190.0, row * 110.0);
            graphScene_->addItem(item);
            if (material == keep) item->setSelected(true);
        }
    }
    pendingFrame_ = true;
    if (isVisible()) frameGraph();
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
    if (pendingFrame_) frameGraph();
}

void MaterialContainerGraphView::resizeEvent(QResizeEvent* event) {
    QGraphicsView::resizeEvent(event);
    if (pendingFrame_) frameGraph();
}

void MaterialContainerGraphView::frameGraph() {
    pendingFrame_ = false;
    if (graphScene_->items().isEmpty()) {
        resetTransform();
        centerOn(0.0, 0.0);
        return;
    }
    const QRectF bounds = graphScene_->itemsBoundingRect().adjusted(-40.0, -40.0, 40.0, 40.0);
    fitInView(bounds, Qt::KeepAspectRatio);
    if (transform().m11() > 1.35) {
        resetTransform();
        centerOn(bounds.center());
    }
}

void MaterialContainerGraphView::wheelEvent(QWheelEvent* event) {
    const double factor = std::pow(1.0018, event->angleDelta().y());
    const double next = transform().m11() * factor;
    if (next < 0.2 || next > 4.0) {
        event->accept();
        return;
    }
    scale(factor, factor);
    event->accept();
}

void MaterialContainerGraphView::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Space) {
        spacePressed_ = true;
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_F) {
        frameGraph();
        event->accept();
        return;
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
    if (event->button() == Qt::MiddleButton) return true;
    if (event->button() == Qt::LeftButton &&
        ((event->modifiers() & Qt::AltModifier) || spacePressed_))
        return true;
    return false;
}

void MaterialContainerGraphView::beginPan(const QPoint& viewPosition) {
    panning_ = true;
    lastPanPoint_ = viewPosition;
    savedDragMode_ = dragMode();
    savedAnchor_ = transformationAnchor();
    setDragMode(QGraphicsView::NoDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setCursor(Qt::ClosedHandCursor);
}

void MaterialContainerGraphView::updatePan(const QPoint& viewPosition) {
    const QPoint delta = viewPosition - lastPanPoint_;
    lastPanPoint_ = viewPosition;
    horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
    verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
}

void MaterialContainerGraphView::endPan() {
    if (!panning_) return;
    panning_ = false;
    setDragMode(savedDragMode_);
    setTransformationAnchor(savedAnchor_);
    unsetCursor();
}

void MaterialContainerGraphView::mousePressEvent(QMouseEvent* event) {
    if (shouldBeginPan(event)) {
        beginPan(event->pos());
        event->accept();
        return;
    }
    QGraphicsView::mousePressEvent(event);
}

void MaterialContainerGraphView::mouseMoveEvent(QMouseEvent* event) {
    if (panning_) {
        updatePan(event->pos());
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
                          "No material nodes in the scene\nAdd a Material node in the Network Editor");
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
    connect(graphView_, &MaterialNetworkGraphView::statusMessage, this, &MaterialNetworkView::statusMessage);
    connect(graphView_, &MaterialNetworkGraphView::selectionChanged, this,
            &MaterialNetworkView::onMaterialXSelectionChanged);

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
    if (inside)
        pathLabel_->setText(QString("Materials  /  %1").arg(currentMaterial_->name()));
    else
        pathLabel_->setText("Materials");
}

void MaterialNetworkView::onContainerSelectionChanged() { emit selectionChanged(); }

void MaterialNetworkView::onMaterialXSelectionChanged() { emit selectionChanged(); }

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
    out.inputs.reserve(node->inputs.size());
    for (const MaterialNetworkGraphView::MtlxInput& input : node->inputs) {
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

}  // namespace sol
