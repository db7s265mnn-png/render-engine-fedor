#include "ui/scene_graph_panel.h"

#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <functional>

#include "ui/theme.h"

namespace sol {

SceneGraphPanel::SceneGraphPanel(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    tree_ = new QTreeWidget(this);
    tree_->setColumnCount(3);
    tree_->setHeaderLabels({"Prim", "Type", "Info"});
    tree_->header()->setStretchLastSection(true);
    tree_->setRootIsDecorated(true);
    tree_->setAlternatingRowColors(false);
    tree_->setUniformRowHeights(true);
    layout->addWidget(tree_, 1);

    summary_ = new QLabel("empty stage", this);
    summary_->setStyleSheet("color: #969aa0;");
    layout->addWidget(summary_, 0);

    connect(tree_, &QTreeWidget::itemSelectionChanged, this, [this] {
        const QList<QTreeWidgetItem*> selected = tree_->selectedItems();
        if (selected.isEmpty()) return;
        emit primSelected(selected.first()->data(0, Qt::UserRole).toString());
    });
}

void SceneGraphPanel::setStage(const StagePtr& stage) {
    tree_->clear();
    if (!stage) {
        summary_->setText("empty stage");
        return;
    }

    QHash<QString, QTreeWidgetItem*> folders;
    std::function<QTreeWidgetItem*(const QString&)> ensureFolder = [&](const QString& path) -> QTreeWidgetItem* {
        if (path.isEmpty()) return nullptr;
        if (folders.contains(path)) return folders[path];
        const int slash = path.lastIndexOf('/');
        const QString parentPath = slash > 0 ? path.left(slash) : QString();
        QTreeWidgetItem* parent = parentPath.isEmpty() ? nullptr : ensureFolder(parentPath);
        auto* item = parent ? new QTreeWidgetItem(parent) : new QTreeWidgetItem(tree_);
        item->setText(0, path.mid(slash + 1));
        item->setText(1, "Scope");
        item->setForeground(1, theme::textDim());
        item->setData(0, Qt::UserRole, path);
        folders.insert(path, item);
        return item;
    };

    long long triangles = 0;
    long long points = 0;
    for (const StagePrim& prim : stage->prims) {
        const int slash = prim.path.lastIndexOf('/');
        QTreeWidgetItem* parent = slash > 0 ? ensureFolder(prim.path.left(slash)) : nullptr;
        auto* item = parent ? new QTreeWidgetItem(parent) : new QTreeWidgetItem(tree_);
        item->setText(0, prim.path.mid(slash + 1));
        item->setText(1, prim.typeName());
        item->setData(0, Qt::UserRole, prim.path);
        if (prim.type == PrimType::Mesh && prim.mesh) {
            item->setText(2, QString("%1 pts / %2 tris")
                                 .arg(prim.mesh->positions.size())
                                 .arg(prim.mesh->triangleCount()));
            triangles += static_cast<long long>(prim.mesh->triangleCount());
            points += static_cast<long long>(prim.mesh->positions.size());
        } else if (prim.type == PrimType::Light) {
            item->setText(2, QString("intensity %1").arg(double(prim.light.intensity), 0, 'g', 3));
        } else if (prim.type == PrimType::Camera) {
            item->setText(2, QString("%1 mm").arg(double(prim.camera.focalLength), 0, 'g', 3));
        }
        if (!prim.active) {
            item->setForeground(0, theme::textDim());
            item->setText(2, item->text(2) + " (pruned)");
        }
        item->setToolTip(0, prim.path + "\nauthored by " + prim.sourceNode);
    }

    tree_->expandAll();
    tree_->resizeColumnToContents(0);
    summary_->setText(QString("%1 prims  |  %2 meshes  |  %3 lights  |  %4 triangles")
                          .arg(stage->prims.size())
                          .arg(stage->countOfType(PrimType::Mesh))
                          .arg(stage->countOfType(PrimType::Light))
                          .arg(triangles));
}

}  // namespace sol
