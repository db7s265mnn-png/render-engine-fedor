#include "ui/material_network_view.h"

#include <QContextMenuEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QKeyEvent>
#include <QLineF>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QScrollBar>
#include <QStyleOptionGraphicsItem>
#include <QWheelEvent>
#include <QXmlStreamReader>
#include <QXmlStreamWriter>
#include <algorithm>
#include <cmath>

#include "io/materialx_graph.h"
#include "nodes/node.h"
#include "nodes/parameter.h"
#include "ui/theme.h"

namespace sol {
namespace {

constexpr qreal kLayoutScale = 80.0;
constexpr qreal kPortRadius = 4.8;
constexpr qreal kPortHitRadius = 11.0;

struct NodeCategoryInfo {
    QString type;
    QColor color;
};

bool isSupportedCategory(const QString& category) {
    return category == "standard_surface" || category == "surfacematerial" || category == "image" ||
           category == "constant" || category == "multiply" || category == "mix" || category == "normalmap";
}

bool isMathCategory(const QString& category) {
    return category == "constant" || category == "multiply" || category == "mix" || category == "normalmap";
}

NodeCategoryInfo categoryInfo(const QString& category) {
    if (category == "image") return {"color3", QColor(42, 132, 132)};
    if (category == "standard_surface") return {"surfaceshader", QColor(189, 116, 45)};
    if (category == "surfacematerial") return {"material", QColor(126, 82, 170)};
    if (category == "normalmap") return {"vector3", QColor(96, 101, 108)};
    if (isMathCategory(category)) return {"color3", QColor(96, 101, 108)};
    return {"color3", QColor(96, 101, 108)};
}

QString defaultTypeForCategory(const QString& category) { return categoryInfo(category).type; }

QColor colorForCategory(const QString& category) { return categoryInfo(category).color; }

bool isConnectableInput(const QString& name, const QString& type) {
    return name != "file" && type != "filename";
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

    int hitInputPort(QPointF scenePosition) const {
        for (int i = 0; i < inputs_.size(); ++i) {
            if (QLineF(scenePosition, inputPortScene(i)).length() <= kPortHitRadius) return i;
        }
        return -1;
    }

    bool hitOutputPort(QPointF scenePosition) const {
        return QLineF(scenePosition, outputPortScene()).length() <= kPortHitRadius;
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

MaterialNetworkNodeItem* outputPortAt(QGraphicsView* view, const QPoint& viewPosition) {
    const QPointF scenePosition = view->mapToScene(viewPosition);
    const QList<QGraphicsItem*> items = view->items(viewPosition);
    for (QGraphicsItem* item : items) {
        if (auto* nodeItem = qgraphicsitem_cast<MaterialNetworkNodeItem*>(item)) {
            if (nodeItem->hitOutputPort(scenePosition)) return nodeItem;
        }
    }
    return nullptr;
}

struct InputHit {
    MaterialNetworkNodeItem* item = nullptr;
    int inputIndex = -1;
};

InputHit inputPortAt(QGraphicsView* view, const QPoint& viewPosition) {
    const QPointF scenePosition = view->mapToScene(viewPosition);
    const QList<QGraphicsItem*> items = view->items(viewPosition);
    for (QGraphicsItem* item : items) {
        if (auto* nodeItem = qgraphicsitem_cast<MaterialNetworkNodeItem*>(item)) {
            const int port = nodeItem->hitInputPort(scenePosition);
            if (port >= 0) return {nodeItem, port};
        }
    }
    return {};
}

void addWire(QGraphicsScene* scene, QPointF from, QPointF to, bool active) {
    auto* wire = scene->addPath(makeWirePath(from, to),
                                QPen(active ? theme::wireActive() : theme::wire(), active ? 2.0 : 1.4,
                                     Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    wire->setZValue(0.0);
}

}  // namespace

MaterialNetworkView::MaterialNetworkView(QWidget* parent) : QGraphicsView(parent) {
    graphScene_ = new QGraphicsScene(this);
    graphScene_->setSceneRect(-700, -450, 1400, 900);
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
    rebuild();
}

void MaterialNetworkView::setMaterialNode(Node* node) {
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

void MaterialNetworkView::ensureMtlxParameter() {
    if (!materialNode_ || materialNode_->findParameter("mtlx")) return;
    materialNode_->addParameter(
        Parameter::makeString("mtlx", "MaterialX XML", QString()).withGroup("MaterialX").withTooltip(
            "Stored MaterialX XML edited by the Material Network view"));
}

QString MaterialNetworkView::defaultDocument() const {
    const QString helperXml = createDefaultMaterialXDocument();
    return helperXml.trimmed().isEmpty() ? fallbackDefaultDocument() : helperXml;
}

void MaterialNetworkView::writeXmlToMaterial(const QString& xml, bool emitEdited) {
    if (!materialNode_) return;
    ensureMtlxParameter();
    suppressMaterialSignal_ = true;
    materialNode_->setParameterValue("mtlx", xml);
    suppressMaterialSignal_ = false;
    if (emitEdited) emit materialEdited(materialNode_);
}

MaterialNetworkView::MtlxNode* MaterialNetworkView::findModelNode(const QString& name) {
    for (MtlxNode& node : graphNodes_) {
        if (node.name == name) return &node;
    }
    return nullptr;
}

const MaterialNetworkView::MtlxNode* MaterialNetworkView::findModelNode(const QString& name) const {
    for (const MtlxNode& node : graphNodes_) {
        if (node.name == name) return &node;
    }
    return nullptr;
}

QString MaterialNetworkView::uniqueNodeName(const QString& baseName) const {
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

void MaterialNetworkView::ensureInput(QVector<MtlxInput>& inputs, const QString& name, const QString& type,
                                      const QString& value) {
    for (MtlxInput& input : inputs) {
        if (input.name == name) {
            if (input.type.isEmpty()) input.type = type;
            return;
        }
    }
    inputs.push_back({name, type, value, {}});
}

QVector<MaterialNetworkView::MtlxInput> MaterialNetworkView::defaultInputsForCategory(const QString& category) {
    QVector<MtlxInput> inputs;
    if (category == "image") {
        inputs.push_back({"file", "filename", {}, {}});
    } else if (category == "constant") {
        inputs.push_back({"value", "color3", "1, 1, 1", {}});
    } else if (category == "multiply") {
        inputs.push_back({"in1", "color3", "1, 1, 1", {}});
        inputs.push_back({"in2", "color3", "1, 1, 1", {}});
    } else if (category == "mix") {
        inputs.push_back({"bg", "color3", "0, 0, 0", {}});
        inputs.push_back({"fg", "color3", "1, 1, 1", {}});
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

void MaterialNetworkView::rebuildFromXml(const QString& xml, bool rewriteRepaired) {
    graphNodes_.clear();
    bool repaired = false;
    bool parsedMaterialX = false;
    int ordinal = 0;

    QXmlStreamReader reader(xml);
    while (!reader.atEnd()) {
        reader.readNext();
        if (!reader.isStartElement()) continue;
        if (reader.name() != QLatin1String("materialx")) {
            reader.skipCurrentElement();
            continue;
        }

        parsedMaterialX = true;
        while (reader.readNextStartElement()) {
            const QString category = reader.name().toString();
            if (!isSupportedCategory(category)) {
                reader.skipCurrentElement();
                continue;
            }

            MtlxNode node;
            node.category = category;
            const QXmlStreamAttributes attrs = reader.attributes();
            node.name = attrs.value("name").toString();
            if (node.name.isEmpty()) {
                node.name = uniqueNodeName(category);
                repaired = true;
            }
            node.type = attrs.value("type").toString();
            if (node.type.isEmpty()) node.type = defaultTypeForCategory(category);

            bool okX = false;
            bool okY = false;
            const qreal x = attrs.value("xpos").toDouble(&okX);
            const qreal y = attrs.value("ypos").toDouble(&okY);
            node.layout = okX && okY ? QPointF(x, y) : defaultLayoutForCategory(category, ordinal);
            if (!okX || !okY) repaired = true;

            while (reader.readNextStartElement()) {
                if (reader.name() == QLatin1String("input")) {
                    MtlxInput input;
                    const QXmlStreamAttributes inputAttrs = reader.attributes();
                    input.name = inputAttrs.value("name").toString();
                    input.type = inputAttrs.value("type").toString();
                    input.value = inputAttrs.value("value").toString();
                    input.nodename = inputAttrs.value("nodename").toString();
                    if (!input.name.isEmpty()) node.inputs.push_back(input);
                    reader.skipCurrentElement();
                } else {
                    reader.skipCurrentElement();
                }
            }

            for (const MtlxInput& input : defaultInputsForCategory(category))
                ensureInput(node.inputs, input.name, input.type, input.value);
            graphNodes_.push_back(node);
            ++ordinal;
        }
    }

    if (!parsedMaterialX || reader.hasError()) {
        graphNodes_.clear();
        repaired = true;
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

void MaterialNetworkView::rebuild() {
    graphScene_->clear();
    previewWire_ = nullptr;
    graphNodes_.clear();

    if (!materialNode_) {
        graphScene_->setSceneRect(-700, -450, 1400, 900);
        viewport()->update();
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
        if (node.category == "image") {
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
                if (sourceItem) addWire(graphScene_, sourceItem->outputPortScene(), targetItem->inputPortScene(visiblePort), true);
            }
            ++visiblePort;
        }
    }

    const QRectF bounds = graphScene_->itemsBoundingRect().adjusted(-180.0, -130.0, 180.0, 130.0);
    graphScene_->setSceneRect(bounds.isEmpty() ? QRectF(-700, -450, 1400, 900) : bounds);
    pendingFrame_ = true;
    if (isVisible() && width() > 50) frameGraph();
}

QString MaterialNetworkView::serializeGraph() const {
    QString xml;
    QXmlStreamWriter writer(&xml);
    writer.setAutoFormatting(true);
    writer.writeStartDocument();
    writer.writeStartElement("materialx");
    writer.writeAttribute("version", "1.38");
    for (const MtlxNode& node : graphNodes_) {
        writer.writeStartElement(node.category);
        writer.writeAttribute("name", node.name);
        writer.writeAttribute("type", node.type.isEmpty() ? defaultTypeForCategory(node.category) : node.type);
        writer.writeAttribute("xpos", QString::number(node.layout.x(), 'f', 3));
        writer.writeAttribute("ypos", QString::number(node.layout.y(), 'f', 3));
        for (const MtlxInput& input : node.inputs) {
            if (input.name.isEmpty()) continue;
            writer.writeEmptyElement("input");
            writer.writeAttribute("name", input.name);
            if (!input.type.isEmpty()) writer.writeAttribute("type", input.type);
            if (!input.nodename.isEmpty()) {
                writer.writeAttribute("nodename", input.nodename);
            } else if (!input.value.isEmpty()) {
                writer.writeAttribute("value", input.value);
            }
        }
        writer.writeEndElement();
    }
    writer.writeEndElement();
    writer.writeEndDocument();
    return xml;
}

void MaterialNetworkView::writeModel(bool emitEdited) { writeXmlToMaterial(serializeGraph(), emitEdited); }

void MaterialNetworkView::showEvent(QShowEvent* event) {
    QGraphicsView::showEvent(event);
    if (pendingFrame_ && width() > 50) frameGraph();
}

void MaterialNetworkView::resizeEvent(QResizeEvent* event) {
    QGraphicsView::resizeEvent(event);
    if (pendingFrame_ && width() > 50) frameGraph();
}

void MaterialNetworkView::frameGraph() {
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

void MaterialNetworkView::wheelEvent(QWheelEvent* event) {
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
    scale(factor, factor);
    event->accept();
}

void MaterialNetworkView::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Space && !event->isAutoRepeat()) {
        spacePressed_ = true;
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

void MaterialNetworkView::keyReleaseEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Space && !event->isAutoRepeat()) {
        spacePressed_ = false;
        event->accept();
        return;
    }
    QGraphicsView::keyReleaseEvent(event);
}

void MaterialNetworkView::contextMenuEvent(QContextMenuEvent* event) {
    lastMousePoint_ = event->pos();
    if (!nodeItemAt(this, event->pos())) {
        showAddNodeMenu(event->pos());
        event->accept();
        return;
    }
    QGraphicsView::contextMenuEvent(event);
}

void MaterialNetworkView::showAddNodeMenu(const QPoint& viewPosition) {
    if (!materialNode_) {
        emit statusMessage("Select a Material node in the Network Editor");
        return;
    }

    QMenu menu(this);
    const QStringList categories = {"image", "constant", "multiply", "mix", "normalmap", "standard_surface",
                                    "surfacematerial"};
    for (const QString& category : categories) {
        QAction* action = menu.addAction(category);
        action->setData(category);
    }

    QAction* chosen = menu.exec(viewport()->mapToGlobal(viewPosition));
    if (!chosen) return;
    addNode(chosen->data().toString(), mapToScene(viewPosition));
}

void MaterialNetworkView::addNode(const QString& category, QPointF scenePosition) {
    if (!materialNode_ || !isSupportedCategory(category)) return;
    MtlxNode node;
    node.category = category;
    node.type = defaultTypeForCategory(category);
    node.name = uniqueNodeName(category == "surfacematerial" ? "surfacematerial" : category);
    node.layout = scenePosition / kLayoutScale;
    node.inputs = defaultInputsForCategory(category);
    graphNodes_.push_back(node);
    writeModel(true);
    rebuild();
    emit statusMessage("Added " + category);
}

void MaterialNetworkView::connectNodes(const QString& sourceName, const QString& targetName, int inputIndex) {
    if (sourceName.isEmpty() || targetName.isEmpty() || sourceName == targetName) return;
    MtlxNode* target = findModelNode(targetName);
    if (!target || inputIndex < 0 || inputIndex >= target->inputs.size()) return;
    target->inputs[inputIndex].nodename = sourceName;
    target->inputs[inputIndex].value.clear();
    writeModel(true);
    rebuild();
    emit statusMessage(QString("Connected %1 to %2.%3").arg(sourceName, targetName, target->inputs[inputIndex].name));
}

void MaterialNetworkView::deleteSelectedNodes() {
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

void MaterialNetworkView::mousePressEvent(QMouseEvent* event) {
    lastMousePoint_ = event->pos();
    mousePressPoint_ = event->pos();
    mouseMovedSincePress_ = false;
    clickImageNode_.clear();

    const bool panButton = event->button() == Qt::MiddleButton ||
                           (event->button() == Qt::LeftButton &&
                            ((event->modifiers() & Qt::AltModifier) || spacePressed_));
    if (panButton) {
        beginPan(event->pos());
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton) {
        if (MaterialNetworkNodeItem* source = outputPortAt(this, event->pos())) {
            beginWire(source->nodeName(), source->outputPortScene());
            event->accept();
            return;
        }
        if (MaterialNetworkNodeItem* item = nodeItemAt(this, event->pos())) {
            if (item->category() == "image") clickImageNode_ = item->nodeName();
        }
    }

    QGraphicsView::mousePressEvent(event);
}

void MaterialNetworkView::mouseMoveEvent(QMouseEvent* event) {
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

void MaterialNetworkView::mouseReleaseEvent(QMouseEvent* event) {
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

void MaterialNetworkView::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && openTextureDialogAt(event->pos())) {
        event->accept();
        return;
    }
    QGraphicsView::mouseDoubleClickEvent(event);
}

bool MaterialNetworkView::openTextureDialogAt(const QPoint& viewPosition) {
    MaterialNetworkNodeItem* item = nodeItemAt(this, viewPosition);
    if (!item || item->category() != "image") return false;
    graphScene_->clearSelection();
    item->setSelected(true);
    chooseTexture(item->nodeName());
    return true;
}

void MaterialNetworkView::chooseTexture(const QString& nodeName) {
    MtlxNode* node = findModelNode(nodeName);
    if (!node || node->category != "image") return;

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

    node->inputs[fileInput].value = path;
    node->inputs[fileInput].nodename.clear();
    writeModel(true);
    rebuild();
    emit statusMessage(QString("%1 file set to %2").arg(nodeName, QFileInfo(path).fileName()));
}

void MaterialNetworkView::syncNodePositions() {
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

void MaterialNetworkView::beginPan(const QPoint& viewPosition) {
    panning_ = true;
    lastPanPoint_ = viewPosition;
    setDragMode(QGraphicsView::NoDrag);
    viewport()->grabMouse();
    viewport()->setCursor(Qt::ClosedHandCursor);
}

void MaterialNetworkView::updatePan(const QPoint& viewPosition) {
    if (viewPosition == lastPanPoint_) return;
    horizontalScrollBar()->setValue(horizontalScrollBar()->value() - (viewPosition.x() - lastPanPoint_.x()));
    verticalScrollBar()->setValue(verticalScrollBar()->value() - (viewPosition.y() - lastPanPoint_.y()));
    lastPanPoint_ = viewPosition;
}

void MaterialNetworkView::endPan() {
    panning_ = false;
    if (QWidget::mouseGrabber() == viewport()) viewport()->releaseMouse();
    setDragMode(QGraphicsView::RubberBandDrag);
    viewport()->unsetCursor();
}

void MaterialNetworkView::beginWire(const QString& sourceName, QPointF sourcePosition) {
    wiring_ = true;
    wireSourceNode_ = sourceName;
    wireSourcePosition_ = sourcePosition;
    setDragMode(QGraphicsView::NoDrag);
    previewWire_ = graphScene_->addPath(makeWirePath(sourcePosition, sourcePosition),
                                        QPen(theme::wireActive(), 1.8, Qt::DashLine, Qt::RoundCap, Qt::RoundJoin));
    previewWire_->setZValue(4.0);
}

void MaterialNetworkView::updateWire(QPointF scenePosition) {
    if (previewWire_) previewWire_->setPath(makeWirePath(wireSourcePosition_, scenePosition));
}

void MaterialNetworkView::endWire(const QPoint& viewPosition) {
    InputHit hit = inputPortAt(this, viewPosition);
    if (previewWire_) {
        graphScene_->removeItem(previewWire_);
        delete previewWire_;
        previewWire_ = nullptr;
    }
    wiring_ = false;
    setDragMode(QGraphicsView::RubberBandDrag);

    if (!hit.item) {
        wireSourceNode_.clear();
        return;
    }
    const int modelInput = hit.item->inputModelIndex(hit.inputIndex);
    connectNodes(wireSourceNode_, hit.item->nodeName(), modelInput);
    wireSourceNode_.clear();
}

void MaterialNetworkView::drawBackground(QPainter* painter, const QRectF& rect) {
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

void MaterialNetworkView::drawForeground(QPainter* painter, const QRectF& rect) {
    QGraphicsView::drawForeground(painter, rect);
    painter->resetTransform();

    QFont font = painter->font();
    if (!materialNode_) {
        font.setPointSizeF(11.0);
        painter->setFont(font);
        painter->setPen(theme::textDim());
        painter->drawText(QRect(0, 0, width(), height()), Qt::AlignCenter,
                          "Select a Material node in the Network Editor");
        return;
    }

    font.setPointSizeF(8.0);
    painter->setFont(font);
    painter->setPen(theme::textDim());
    painter->drawText(QRect(8, height() - 22, width() - 16, 18), Qt::AlignLeft,
                      "Tab: add node   LMB wire   MMB pan   Del: delete   image dbl-click: file");
}

}  // namespace sol
