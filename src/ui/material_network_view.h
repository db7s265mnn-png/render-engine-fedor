// MaterialX-inspired subnet view for editing a material node's texture inputs.
#pragma once

#include <QGraphicsView>
#include <QMetaObject>
#include <QPoint>

namespace sol {

class Node;

class MaterialNetworkView : public QGraphicsView {
    Q_OBJECT

public:
    explicit MaterialNetworkView(QWidget* parent = nullptr);

    void setMaterialNode(Node* node);

signals:
    void materialEdited(sol::Node* node);
    void statusMessage(const QString& message);

protected:
    void showEvent(QShowEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void drawBackground(QPainter* painter, const QRectF& rect) override;
    void drawForeground(QPainter* painter, const QRectF& rect) override;

private:
    void rebuild();
    void frameGraph();
    bool openTextureDialogAt(const QPoint& viewPosition);
    void chooseTexture(const QString& parameterName, const QString& inputName);
    void beginPan(const QPoint& viewPosition);
    void updatePan(const QPoint& viewPosition);
    void endPan();

    Node* materialNode_ = nullptr;
    QGraphicsScene* graphScene_ = nullptr;
    QMetaObject::Connection materialChangedConnection_;
    bool pendingFrame_ = false;
    bool panning_ = false;
    QPoint lastPanPoint_;
};

}  // namespace sol
