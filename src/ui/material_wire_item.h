// Selectable bezier wire between MaterialX ports (right-click to disconnect).
#pragma once

#include <QGraphicsPathItem>
#include <QPainterPath>
#include <QString>

namespace sol {

class MaterialWireItem : public QGraphicsPathItem {
public:
    enum { Type = UserType + 73 };

    MaterialWireItem(QString sourceNodeName, QString targetNodeName, QString inputName,
                     QGraphicsItem* parent = nullptr);

    int type() const override { return Type; }
    const QString& sourceNodeName() const { return sourceNodeName_; }
    const QString& targetNodeName() const { return targetNodeName_; }
    const QString& inputName() const { return inputName_; }

    void setWirePath(const QPainterPath& path);

    QPainterPath shape() const override;
    QRectF boundingRect() const override;

private:
    QString sourceNodeName_;
    QString targetNodeName_;
    QString inputName_;
};

}  // namespace sol
