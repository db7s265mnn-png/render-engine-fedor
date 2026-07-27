// Scene graph tree of the cooked stage, mirroring the Solaris prim hierarchy.
// Also lists material container nodes from the LOPs network (not MaterialX guts).
#pragma once

#include <QStringList>
#include <QWidget>

#include "nodes/stage.h"

class QTreeWidget;
class QLabel;

namespace sol {

class SceneGraphPanel : public QWidget {
    Q_OBJECT

public:
    explicit SceneGraphPanel(QWidget* parent = nullptr);

    void setStage(const StagePtr& stage, const QStringList& materialContainers = {});
    QString selectedPath() const;
    QString selectedSourceNode() const;
    // Highlight a prim authored by sourceNode (or clear when empty). Does not emit itemSelected.
    void selectBySourceNode(const QString& sourceNode);

signals:
    // Fired when the user picks a prim or material container.
    // sourceNode is the authoring LOP node name when available.
    void itemSelected(const QString& path, const QString& sourceNode);

private:
    void emitCurrentSelection();

    QTreeWidget* tree_ = nullptr;
    QLabel* summary_ = nullptr;
    QString pendingSelectPath_;
};

}  // namespace sol
