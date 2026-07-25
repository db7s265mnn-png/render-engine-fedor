#include "ui/scene_graph_panel.h"

#include <QApplication>
#include <QClipboard>
#include <QDrag>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QKeyEvent>
#include <QLabel>
#include <QMenu>
#include <QMimeData>
#include <QMouseEvent>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <functional>

#include "ui/theme.h"

namespace sol {
namespace {

constexpr const char* kPrimPathMime = "application/x-fedor-prim-path";

class PrimTreeWidget : public QTreeWidget {
public:
    using QTreeWidget::QTreeWidget;

protected:
    void keyPressEvent(QKeyEvent* event) override {
        if (event->matches(QKeySequence::Copy)) {
            copySelectedPrimPath();
            event->accept();
            return;
        }
        QTreeWidget::keyPressEvent(event);
    }

    void mousePressEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton) dragStartPos_ = event->pos();
        QTreeWidget::mousePressEvent(event);
    }

    void mouseMoveEvent(QMouseEvent* event) override {
        if (!(event->buttons() & Qt::LeftButton) || dragStartPos_.isNull()) {
            QTreeWidget::mouseMoveEvent(event);
            return;
        }
        if ((event->pos() - dragStartPos_).manhattanLength() < QApplication::startDragDistance()) {
            QTreeWidget::mouseMoveEvent(event);
            return;
        }

        QTreeWidgetItem* item = currentItem();
        if (!item) {
            QTreeWidget::mouseMoveEvent(event);
            return;
        }
        const QString path = item->data(0, Qt::UserRole).toString();
        if (path.isEmpty()) {
            QTreeWidget::mouseMoveEvent(event);
            return;
        }

        auto* mime = new QMimeData();
        mime->setText(path);
        mime->setData(QByteArray(kPrimPathMime), path.toUtf8());

        auto* drag = new QDrag(this);
        drag->setMimeData(mime);
        drag->exec(Qt::CopyAction);
        dragStartPos_ = QPoint();
    }

public:
    void copySelectedPrimPath() {
        const QList<QTreeWidgetItem*> selected = selectedItems();
        if (selected.isEmpty()) return;
        const QString path = selected.first()->data(0, Qt::UserRole).toString();
        if (path.isEmpty()) return;
        QApplication::clipboard()->setText(path);
    }

private:
    QPoint dragStartPos_;
};

}  // namespace

SceneGraphPanel::SceneGraphPanel(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    auto* tree = new PrimTreeWidget(this);
    tree_ = tree;
    tree_->setColumnCount(3);
    tree_->setHeaderLabels({"Prim", "Type", "Info"});
    tree_->header()->setStretchLastSection(true);
    tree_->setRootIsDecorated(true);
    tree_->setAlternatingRowColors(false);
    tree_->setUniformRowHeights(true);
    tree_->setSelectionMode(QAbstractItemView::SingleSelection);
    // Drag is handled manually so the full prim path (not the leaf name) is sent.
    tree_->setDragEnabled(false);
    tree_->setFocusPolicy(Qt::StrongFocus);
    layout->addWidget(tree_, 1);

    summary_ = new QLabel("empty stage", this);
    summary_->setStyleSheet("color: #969aa0;");
    layout->addWidget(summary_, 0);

    connect(tree_, &QTreeWidget::itemSelectionChanged, this, [this] {
        const QList<QTreeWidgetItem*> selected = tree_->selectedItems();
        if (selected.isEmpty()) return;
        emit primSelected(selected.first()->data(0, Qt::UserRole).toString());
    });

    tree_->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(tree_, &QWidget::customContextMenuRequested, this, [this, tree](const QPoint& pos) {
        QTreeWidgetItem* item = tree_->itemAt(pos);
        if (!item) return;
        tree_->setCurrentItem(item);
        QMenu menu(this);
        menu.addAction("Copy Prim Path", QKeySequence::Copy, tree, &PrimTreeWidget::copySelectedPrimPath);
        menu.exec(tree_->viewport()->mapToGlobal(pos));
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
        item->setFlags(item->flags() | Qt::ItemIsDragEnabled);
        item->setToolTip(0, path + "\nCtrl+C or drag into Material → Assign To");
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
        item->setFlags(item->flags() | Qt::ItemIsDragEnabled);
        item->setToolTip(0, prim.path + "\nauthored by " + prim.sourceNode +
                                "\nCtrl+C or drag into Material → Assign To");
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
