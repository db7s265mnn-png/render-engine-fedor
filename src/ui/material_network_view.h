// Solaris-style MaterialX graph view backed by a material node's "mtlx" XML.
#pragma once

#include <QGraphicsView>
#include <QMetaObject>
#include <QPoint>
#include <QString>
#include <QVector>

#include "nodes/node.h"

class QGraphicsPathItem;
class QGraphicsScene;

namespace sol {

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
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void contextMenuEvent(QContextMenuEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void drawBackground(QPainter* painter, const QRectF& rect) override;
    void drawForeground(QPainter* painter, const QRectF& rect) override;

private:
    struct MtlxInput {
        QString name;
        QString type;
        QString value;
        QString nodename;
    };

    struct MtlxNode {
        QString name;
        QString category;
        QString type;
        QPointF layout;
        QVector<MtlxInput> inputs;
    };

    void rebuild();
    void rebuildFromXml(const QString& xml, bool rewriteRepaired);
    QString serializeGraph() const;
    QString defaultDocument() const;
    void ensureMtlxParameter();
    void writeXmlToMaterial(const QString& xml, bool emitEdited);
    void writeModel(bool emitEdited);
    void frameGraph();
    void showAddNodeMenu(const QPoint& viewPosition);
    void addNode(const QString& category, QPointF scenePosition);
    void connectNodes(const QString& sourceName, const QString& targetName, int inputIndex);
    void deleteSelectedNodes();
    bool openTextureDialogAt(const QPoint& viewPosition);
    void chooseTexture(const QString& nodeName);
    void syncNodePositions();
    void beginPan(const QPoint& viewPosition);
    void updatePan(const QPoint& viewPosition);
    void endPan();
    void beginWire(const QString& sourceName, QPointF sourcePosition);
    void updateWire(QPointF scenePosition);
    void endWire(const QPoint& viewPosition);
    QString uniqueNodeName(const QString& baseName) const;
    static void ensureInput(QVector<MtlxInput>& inputs, const QString& name, const QString& type,
                            const QString& value = QString());
    static QVector<MtlxInput> defaultInputsForCategory(const QString& category);
    MtlxNode* findModelNode(const QString& name);
    const MtlxNode* findModelNode(const QString& name) const;

    Node* materialNode_ = nullptr;
    QGraphicsScene* graphScene_ = nullptr;
    QMetaObject::Connection materialChangedConnection_;
    QVector<MtlxNode> graphNodes_;
    QGraphicsPathItem* previewWire_ = nullptr;
    bool pendingFrame_ = false;
    bool panning_ = false;
    bool spacePressed_ = false;
    bool wiring_ = false;
    bool suppressMaterialSignal_ = false;
    bool mouseMovedSincePress_ = false;
    QPoint lastPanPoint_;
    QPoint lastMousePoint_;
    QPoint mousePressPoint_;
    QString wireSourceNode_;
    QPointF wireSourcePosition_;
    QString clickImageNode_;
};

}  // namespace sol
