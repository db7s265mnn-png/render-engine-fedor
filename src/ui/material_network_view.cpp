#include "ui/material_network_view.h"

#include <QFileDialog>
#include <QFileInfo>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QScrollBar>
#include <QStyleOptionGraphicsItem>
#include <QWheelEvent>
#include <algorithm>
#include <array>
#include <cmath>

#include "nodes/node.h"
#include "nodes/parameter.h"
#include "ui/theme.h"

namespace sol {
namespace {

struct TextureSlot {
    const char* inputName;
    const char* parameterName;
    const char* label;
};

constexpr std::array<TextureSlot, 7> kTextureSlots{{
    {"base_color", "basecolor_texture", "Base Color"},
    {"roughness", "roughness_texture", "Roughness"},
    {"metallic", "metallic_texture", "Metallic"},
    {"opacity", "opacity_texture", "Opacity"},
    {"emission_color", "emission_texture", "Emission"},
    {"normal", "normal_texture", "Normal"},
    {"subsurface_color", "subsurface_texture", "Subsurface"},
}};

enum class MaterialItemRole { Surface, StandardSurface, Image };

QPainterPath makeMaterialWirePath(QPointF from, QPointF to) {
    QPainterPath path(from);
    if (std::abs(to.x() - from.x()) > std::abs(to.y() - from.y())) {
        const qreal dx = std::max(qreal(36.0), std::abs(to.x() - from.x()) * 0.5);
        path.cubicTo(from + QPointF(dx, 0.0), to - QPointF(dx, 0.0), to);
    } else {
        const qreal dy = std::max(qreal(34.0), std::abs(to.y() - from.y()) * 0.45);
        path.cubicTo(from + QPointF(0.0, dy), to - QPointF(0.0, dy), to);
    }
    return path;
}

QRectF centeredRect(qreal width, qreal height) {
    return QRectF(-width * 0.5, -height * 0.5, width, height);
}

class MaterialNetworkNodeItem : public QGraphicsItem {
public:
    enum { Type = UserType + 40 };

    MaterialNetworkNodeItem(MaterialItemRole role, QString name, QString typeName)
        : role_(role), name_(std::move(name)), typeName_(std::move(typeName)) {
        setFlag(ItemIsSelectable, true);
        setZValue(1.0);
    }

    int type() const override { return Type; }

    MaterialItemRole role() const { return role_; }
    const QString& parameterName() const { return parameterName_; }
    const QString& inputName() const { return inputName_; }
    bool parameterExists() const { return parameterExists_; }

    void setTextureSlot(const TextureSlot& slot, bool parameterExists, QString filePath) {
        inputName_ = QString::fromUtf8(slot.inputName);
        parameterName_ = QString::fromUtf8(slot.parameterName);
        slotLabel_ = QString::fromUtf8(slot.label);
        parameterExists_ = parameterExists;
        filePath_ = std::move(filePath);
        if (parameterExists_) setCursor(Qt::PointingHandCursor);
    }

    QRectF boundingRect() const override { return bodyRect().adjusted(-8.0, -8.0, 8.0, 8.0); }

    QPainterPath shape() const override {
        QPainterPath path;
        path.addRoundedRect(bodyRect(), 5.0, 5.0);
        if (hasInputPort()) path.addEllipse(inputPortPosition() - pos(), kPortHitRadius, kPortHitRadius);
        if (hasOutputPort()) path.addEllipse(outputPortPosition() - pos(), kPortHitRadius, kPortHitRadius);
        if (role_ == MaterialItemRole::StandardSurface) {
            for (int i = 0; i < int(kTextureSlots.size()); ++i)
                path.addEllipse(inputPortPosition(i) - pos(), kPortHitRadius, kPortHitRadius);
        }
        return path;
    }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) override {
        painter->setRenderHint(QPainter::Antialiasing, true);

        const QRectF body = bodyRect();
        painter->setPen(Qt::NoPen);
        painter->setBrush(QColor(0, 0, 0, 72));
        painter->drawRoundedRect(body.translated(2.0, 3.0), 5.0, 5.0);

        QLinearGradient gradient(body.topLeft(), body.bottomLeft());
        gradient.setColorAt(0.0, theme::panelLight().lighter(role_ == MaterialItemRole::Image ? 112 : 108));
        gradient.setColorAt(1.0, theme::panel().darker(112));
        painter->setBrush(gradient);

        QPen border(parameterExists_ || role_ != MaterialItemRole::Image ? QColor(20, 21, 24)
                                                                         : theme::gridLine().lighter(130),
                    isSelected() ? 2.0 : 1.2);
        if (isSelected()) border.setColor(theme::selection());
        if (role_ == MaterialItemRole::Image && !parameterExists_) border.setStyle(Qt::DashLine);
        painter->setPen(border);
        painter->drawRoundedRect(body, 5.0, 5.0);

        QPainterPath clip;
        clip.addRoundedRect(body, 5.0, 5.0);
        painter->save();
        painter->setClipPath(clip);
        painter->setPen(Qt::NoPen);
        painter->setBrush(headerColor());
        painter->drawRect(QRectF(body.left(), body.top(), body.width(), 10.0));
        painter->restore();

        paintLabels(painter, body);
        paintPorts(painter);
    }

    QPointF inputPortPosition() const { return pos() + QPointF(0.0, -bodyRect().height() * 0.5 - 2.0); }

    QPointF inputPortPosition(int index) const {
        if (role_ != MaterialItemRole::StandardSurface) return inputPortPosition();
        const qreal top = -108.0;
        return pos() + QPointF(-bodyRect().width() * 0.5 - 2.0, top + qreal(index) * 36.0);
    }

    QPointF outputPortPosition() const {
        if (role_ == MaterialItemRole::Image) return pos() + QPointF(bodyRect().width() * 0.5 + 2.0, 0.0);
        return pos() + QPointF(0.0, bodyRect().height() * 0.5 + 2.0);
    }

private:
    static constexpr qreal kPortRadius = 5.2;
    static constexpr qreal kPortHitRadius = 15.0;

    QRectF bodyRect() const {
        switch (role_) {
            case MaterialItemRole::Surface: return centeredRect(112.0, 50.0);
            case MaterialItemRole::StandardSurface: return centeredRect(154.0, 300.0);
            case MaterialItemRole::Image: return centeredRect(118.0, 38.0);
        }
        return centeredRect(100.0, 44.0);
    }

    QColor headerColor() const {
        switch (role_) {
            case MaterialItemRole::Surface: return QColor(68, 92, 118);
            case MaterialItemRole::StandardSurface: return QColor(138, 106, 63);
            case MaterialItemRole::Image:
                if (!parameterExists_) return theme::gridLine().lighter(120);
                return filePath_.trimmed().isEmpty() ? QColor(82, 91, 104) : QColor(68, 116, 138);
        }
        return theme::panelLight();
    }

    bool hasInputPort() const { return role_ == MaterialItemRole::Surface; }
    bool hasOutputPort() const {
        return role_ == MaterialItemRole::StandardSurface || role_ == MaterialItemRole::Image;
    }

    void paintLabels(QPainter* painter, const QRectF& body) const {
        QFont nameFont = painter->font();
        nameFont.setPointSizeF(8.3);
        nameFont.setBold(true);
        painter->setFont(nameFont);
        painter->setPen(parameterExists_ || role_ != MaterialItemRole::Image ? theme::text() : theme::textDim());
        const QRectF nameRect = body.adjusted(8.0, 8.0, -8.0, -body.height() + 27.0);
        painter->drawText(nameRect, Qt::AlignLeft | Qt::AlignVCenter,
                          QFontMetrics(nameFont).elidedText(name_, Qt::ElideRight, int(nameRect.width())));

        QFont typeFont = painter->font();
        typeFont.setBold(false);
        typeFont.setPointSizeF(7.0);
        painter->setFont(typeFont);
        painter->setPen(theme::textDim());
        QRectF typeRect(body.left() + 8.0, body.top() + 25.0, body.width() - 16.0, 14.0);

        QString subtitle = typeName_;
        if (role_ == MaterialItemRole::Image) {
            if (!parameterExists_)
                subtitle = "missing " + parameterName_;
            else if (filePath_.trimmed().isEmpty())
                subtitle = "click to choose";
            else
                subtitle = QFileInfo(filePath_).fileName();
        }
        painter->drawText(typeRect, Qt::AlignLeft | Qt::AlignVCenter,
                          QFontMetrics(typeFont).elidedText(subtitle, Qt::ElideRight, int(typeRect.width())));

        if (role_ == MaterialItemRole::StandardSurface) {
            for (int i = 0; i < int(kTextureSlots.size()); ++i) {
                const QPointF port = inputPortPosition(i) - pos();
                const QString label = QString::fromUtf8(kTextureSlots[size_t(i)].inputName);
                const QRectF labelRect(port.x() + 10.0, port.y() - 8.0, body.width() - 16.0, 16.0);
                painter->drawText(labelRect, Qt::AlignLeft | Qt::AlignVCenter,
                                  QFontMetrics(typeFont).elidedText(label, Qt::ElideRight,
                                                                    int(labelRect.width())));
            }
        }
    }

    void paintPorts(QPainter* painter) const {
        auto drawPort = [painter](QPointF local, bool connected) {
            painter->setPen(Qt::NoPen);
            painter->setBrush(QColor(120, 170, 255, connected ? 58 : 35));
            painter->drawEllipse(local, kPortRadius + 3.0, kPortRadius + 3.0);
            painter->setPen(QPen(QColor(20, 21, 24), 1.0));
            painter->setBrush(connected ? theme::wireActive() : QColor(120, 125, 133));
            painter->drawEllipse(local, kPortRadius, kPortRadius);
        };

        if (role_ == MaterialItemRole::Surface) drawPort(inputPortPosition() - pos(), true);
        if (role_ == MaterialItemRole::StandardSurface) {
            for (int i = 0; i < int(kTextureSlots.size()); ++i) drawPort(inputPortPosition(i) - pos(), false);
            drawPort(outputPortPosition() - pos(), true);
        }
        if (role_ == MaterialItemRole::Image) drawPort(outputPortPosition() - pos(), !filePath_.trimmed().isEmpty());
    }

    MaterialItemRole role_;
    QString name_;
    QString typeName_;
    QString slotLabel_;
    QString inputName_;
    QString parameterName_;
    QString filePath_;
    bool parameterExists_ = true;
};

MaterialNetworkNodeItem* materialItemAt(QGraphicsView* view, QPoint viewPosition) {
    const QList<QGraphicsItem*> items = view->items(viewPosition);
    for (QGraphicsItem* item : items) {
        if (auto* nodeItem = qgraphicsitem_cast<MaterialNetworkNodeItem*>(item)) return nodeItem;
    }
    return nullptr;
}

void addWire(QGraphicsScene* scene, QPointF from, QPointF to, bool active) {
    auto* wire = scene->addPath(makeMaterialWirePath(from, to),
                                QPen(active ? theme::wireActive() : theme::wire(), active ? 2.0 : 1.6,
                                     Qt::SolidLine, Qt::RoundCap));
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
    rebuild();
}

void MaterialNetworkView::setMaterialNode(Node* node) {
    if (node && node->typeName() != "material") node = nullptr;
    if (materialNode_ == node) return;

    if (materialChangedConnection_) disconnect(materialChangedConnection_);
    materialChangedConnection_ = {};
    materialNode_ = node;

    if (materialNode_) {
        materialChangedConnection_ =
            connect(materialNode_, &Node::parameterChanged, this,
                    [this](Node* node, const QString&) {
                        if (node == materialNode_) rebuild();
                    });
    }

    rebuild();
}

void MaterialNetworkView::rebuild() {
    graphScene_->clear();

    if (!materialNode_) {
        graphScene_->setSceneRect(-700, -450, 1400, 900);
        viewport()->update();
        return;
    }

    auto* standard = new MaterialNetworkNodeItem(MaterialItemRole::StandardSurface, "standard_surface",
                                                "MaterialX");
    standard->setPos(80.0, 0.0);
    graphScene_->addItem(standard);

    auto* surface = new MaterialNetworkNodeItem(MaterialItemRole::Surface, "surface", "output");
    surface->setPos(80.0, 210.0);
    graphScene_->addItem(surface);
    addWire(graphScene_, standard->outputPortPosition(), surface->inputPortPosition(), true);

    for (int i = 0; i < int(kTextureSlots.size()); ++i) {
        const TextureSlot& slot = kTextureSlots[size_t(i)];
        const Parameter* parameter = materialNode_->findParameter(QString::fromUtf8(slot.parameterName));
        const QString filePath = parameter ? parameter->toString() : QString();

        auto* image =
            new MaterialNetworkNodeItem(MaterialItemRole::Image, "image", QString::fromUtf8(slot.label));
        image->setTextureSlot(slot, parameter != nullptr, filePath);
        image->setPos(-220.0, standard->inputPortPosition(i).y());
        graphScene_->addItem(image);

        if (parameter && !filePath.trimmed().isEmpty())
            addWire(graphScene_, image->outputPortPosition(), standard->inputPortPosition(i), true);
    }

    const QRectF bounds = graphScene_->itemsBoundingRect().adjusted(-160.0, -120.0, 180.0, 120.0);
    graphScene_->setSceneRect(bounds);
    pendingFrame_ = true;
    if (isVisible() && width() > 50) frameGraph();
}

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
    fitInView(bounds.adjusted(-70.0, -70.0, 70.0, 70.0), Qt::KeepAspectRatio);
    const double scale = transform().m11();
    if (scale > 1.0 || scale < 0.55) {
        resetTransform();
        this->scale(std::clamp(scale, 0.55, 1.0), std::clamp(scale, 0.55, 1.0));
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

void MaterialNetworkView::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::MiddleButton ||
        (event->button() == Qt::LeftButton && (event->modifiers() & Qt::AltModifier))) {
        beginPan(event->pos());
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton && openTextureDialogAt(event->pos())) {
        event->accept();
        return;
    }

    QGraphicsView::mousePressEvent(event);
    if (auto* item = materialItemAt(this, event->pos())) {
        if (item->role() == MaterialItemRole::StandardSurface) emit statusMessage("standard_surface");
    }
}

void MaterialNetworkView::mouseMoveEvent(QMouseEvent* event) {
    if (panning_) {
        updatePan(event->pos());
        event->accept();
        return;
    }
    QGraphicsView::mouseMoveEvent(event);
}

void MaterialNetworkView::mouseReleaseEvent(QMouseEvent* event) {
    if (panning_) {
        endPan();
        event->accept();
        return;
    }
    QGraphicsView::mouseReleaseEvent(event);
}

void MaterialNetworkView::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && openTextureDialogAt(event->pos())) {
        event->accept();
        return;
    }
    QGraphicsView::mouseDoubleClickEvent(event);
}

bool MaterialNetworkView::openTextureDialogAt(const QPoint& viewPosition) {
    MaterialNetworkNodeItem* item = materialItemAt(this, viewPosition);
    if (!item || item->role() != MaterialItemRole::Image) return false;

    graphScene_->clearSelection();
    item->setSelected(true);
    if (!item->parameterExists()) {
        emit statusMessage("Material parameter is not available: " + item->parameterName());
        return true;
    }

    chooseTexture(item->parameterName(), item->inputName());
    return true;
}

void MaterialNetworkView::chooseTexture(const QString& parameterName, const QString& inputName) {
    if (!materialNode_) return;
    Parameter* parameter = materialNode_->findParameter(parameterName);
    if (!parameter) {
        emit statusMessage("Material parameter is not available: " + parameterName);
        return;
    }

    const QString filter = parameter->fileFilter.isEmpty()
                               ? QString("Images (*.png *.jpg *.jpeg *.exr *.hdr *.tif *.tiff *.bmp *.webp);;"
                                         "All Files (*)")
                               : parameter->fileFilter;
    const QString path =
        QFileDialog::getOpenFileName(this, "Choose texture for " + inputName, parameter->toString(), filter);
    if (path.isEmpty()) return;

    materialNode_->setParameterValue(parameterName, path);
    emit materialEdited(materialNode_);
    emit statusMessage(QString("%1 texture set to %2").arg(inputName, QFileInfo(path).fileName()));
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
    const QPointF delta = mapToScene(viewPosition) - mapToScene(lastPanPoint_);
    translate(delta.x(), delta.y());
    lastPanPoint_ = viewPosition;
}

void MaterialNetworkView::endPan() {
    panning_ = false;
    if (QWidget::mouseGrabber() == viewport()) viewport()->releaseMouse();
    setDragMode(QGraphicsView::RubberBandDrag);
    viewport()->unsetCursor();
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
                      "Material subnet   click image: choose texture   MMB/Alt+LMB: pan   Wheel: zoom");
}

}  // namespace sol
