// Scene graph tree of the cooked stage, mirroring the Solaris prim hierarchy.
#pragma once

#include <QWidget>

#include "nodes/stage.h"

class QTreeWidget;
class QLabel;

namespace sol {

class SceneGraphPanel : public QWidget {
    Q_OBJECT

public:
    explicit SceneGraphPanel(QWidget* parent = nullptr);

    void setStage(const StagePtr& stage);

signals:
    void primSelected(const QString& path);

private:
    QTreeWidget* tree_ = nullptr;
    QLabel* summary_ = nullptr;
};

}  // namespace sol
