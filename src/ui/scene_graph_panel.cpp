#include "ui/scene_graph_panel.h"

#include <QApplication>
#include <QAbstractItemView>
#include <QClipboard>
#include <QDrag>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QSizePolicy>
#include <QKeyEvent>
#include <QLabel>
#include <QMenu>
#include <QMimeData>
#include <QMouseEvent>
#include <QSignalBlocker>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <functional>

#include "ui/theme.h"

namespace sol {
namespace {

constexpr const char* kPrimPathMime = "application/x-fedor-prim-path";
constexpr int kRolePath = Qt::UserRole;
constexpr int kRoleSourceNode = Qt::UserRole + 1;

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
        if (event->key() == Qt::Key_F) {
            if (auto* panel = qobject_cast<SceneGraphPanel*>(parentWidget())) {
                panel->requestFrameSelected();
                event->accept();
                return;
            }
        }
        if (event->key() == Qt::Key_H || event->key() == Qt::Key_Home) {
            if (auto* panel = qobject_cast<SceneGraphPanel*>(parentWidget())) {
                panel->requestFrameAll();
                event->accept();
                return;
            }
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
        const QString path = item->data(0, kRolePath).toString();
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
        const QString path = selected.first()->data(0, kRolePath).toString();
        if (path.isEmpty()) return;
        QApplication::clipboard()->setText(path);
    }

private:
    QPoint dragStartPos_;
};

QTreeWidgetItem* findItemByPath(QTreeWidget* tree, const QString& path) {
    if (!tree || path.isEmpty()) return nullptr;
    std::function<QTreeWidgetItem*(QTreeWidgetItem*)> walk = [&](QTreeWidgetItem* item) -> QTreeWidgetItem* {
        if (!item) return nullptr;
        if (item->data(0, kRolePath).toString() == path) return item;
        for (int i = 0; i < item->childCount(); ++i) {
            if (QTreeWidgetItem* hit = walk(item->child(i))) return hit;
        }
        return nullptr;
    };
    for (int i = 0; i < tree->topLevelItemCount(); ++i) {
        if (QTreeWidgetItem* hit = walk(tree->topLevelItem(i))) return hit;
    }
    return nullptr;
}

QTreeWidgetItem* findItemBySourceNode(QTreeWidget* tree, const QString& sourceNode) {
    if (!tree || sourceNode.isEmpty()) return nullptr;
    std::function<QTreeWidgetItem*(QTreeWidgetItem*)> walk = [&](QTreeWidgetItem* item) -> QTreeWidgetItem* {
        if (!item) return nullptr;
        if (item->data(0, kRoleSourceNode).toString() == sourceNode) return item;
        for (int i = 0; i < item->childCount(); ++i) {
            if (QTreeWidgetItem* hit = walk(item->child(i))) return hit;
        }
        return nullptr;
    };
    for (int i = 0; i < tree->topLevelItemCount(); ++i) {
        if (QTreeWidgetItem* hit = walk(tree->topLevelItem(i))) return hit;
    }
    return nullptr;
}

}  // namespace

SceneGraphPanel::SceneGraphPanel(QWidget* parent) : QWidget(parent) {
    setMinimumWidth(80);
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    auto* tree = new PrimTreeWidget(this);
    tree_ = tree;
    tree_->setColumnCount(3);
    tree_->setHeaderLabels({"Prim", "Type", "Info"});
    tree_->header()->setStretchLastSection(true);
    tree_->header()->setMinimumSectionSize(24);
    tree_->header()->setSectionResizeMode(QHeaderView::Interactive);
    tree_->header()->setDefaultSectionSize(72);
    tree_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    tree_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
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

    connect(tree_, &QTreeWidget::itemSelectionChanged, this, [this] { emitCurrentSelection(); });

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

QString SceneGraphPanel::selectedPath() const {
    const QList<QTreeWidgetItem*> selected = tree_->selectedItems();
    if (selected.isEmpty()) return {};
    return selected.first()->data(0, kRolePath).toString();
}

QString SceneGraphPanel::selectedSourceNode() const {
    const QList<QTreeWidgetItem*> selected = tree_->selectedItems();
    if (selected.isEmpty()) return {};
    return selected.first()->data(0, kRoleSourceNode).toString();
}

void SceneGraphPanel::emitCurrentSelection() {
    const QList<QTreeWidgetItem*> selected = tree_->selectedItems();
    if (selected.isEmpty()) return;
    emit itemSelected(selected.first()->data(0, kRolePath).toString(),
                      selected.first()->data(0, kRoleSourceNode).toString());
}

void SceneGraphPanel::selectBySourceNode(const QString& sourceNode) {
    const QSignalBlocker blocker(tree_);
    tree_->clearSelection();
    if (sourceNode.isEmpty()) {
        tree_->setCurrentItem(nullptr);
        return;
    }
    if (QTreeWidgetItem* item = findItemBySourceNode(tree_, sourceNode)) {
        tree_->setCurrentItem(item);
        item->setSelected(true);
        tree_->scrollToItem(item, QAbstractItemView::EnsureVisible);
    }
}

void SceneGraphPanel::setStage(const StagePtr& stage, const QStringList& materialContainers) {
    const QString keepPath = pendingSelectPath_.isEmpty() ? selectedPath() : pendingSelectPath_;
    pendingSelectPath_.clear();

    const QSignalBlocker blocker(tree_);
    tree_->clear();
    if (!stage && materialContainers.isEmpty()) {
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
        item->setData(0, kRolePath, path);
        item->setFlags(item->flags() | Qt::ItemIsDragEnabled);
        item->setToolTip(0, path + "\nCtrl+C or drag into Material → Assign To");
        folders.insert(path, item);
        return item;
    };

    long long triangles = 0;
    long long polygons = 0;
    long long points = 0;
    int primCount = 0;
    if (stage) {
        primCount = int(stage->prims.size());
        for (const StagePrim& prim : stage->prims) {
            const int slash = prim.path.lastIndexOf('/');
            QTreeWidgetItem* parent = slash > 0 ? ensureFolder(prim.path.left(slash)) : nullptr;
            auto* item = parent ? new QTreeWidgetItem(parent) : new QTreeWidgetItem(tree_);
            item->setText(0, prim.path.mid(slash + 1));
            item->setText(1, prim.typeName());
            item->setData(0, kRolePath, prim.path);
            item->setData(0, kRoleSourceNode, prim.sourceNode);
            item->setFlags(item->flags() | Qt::ItemIsDragEnabled);
            item->setToolTip(0, prim.path + "\nauthored by " + prim.sourceNode +
                                    "\nCtrl+C or drag into Material → Assign To");
            if (prim.type == PrimType::Mesh && prim.mesh) {
                const auto tris = prim.mesh->triangleCount();
                const auto faces = prim.mesh->faceCount();
                item->setText(2, QString("%1 pts / %2 tris / %3 polys")
                                     .arg(prim.mesh->positions.size())
                                     .arg(tris)
                                     .arg(faces));
                triangles += static_cast<long long>(tris);
                polygons += static_cast<long long>(faces);
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
    }

    // Material containers from the LOPs network (not MaterialX internals).
    if (!materialContainers.isEmpty()) {
        QTreeWidgetItem* materialsRoot = ensureFolder("/materials");
        materialsRoot->setText(1, "Materials");
        materialsRoot->setText(2, QString("%1 containers").arg(materialContainers.size()));
        for (const QString& name : materialContainers) {
            if (name.isEmpty()) continue;
            auto* item = new QTreeWidgetItem(materialsRoot);
            item->setText(0, name);
            item->setText(1, "Material");
            item->setText(2, "container");
            item->setData(0, kRolePath, "/materials/" + name);
            item->setData(0, kRoleSourceNode, name);
            item->setToolTip(0, "Material container node: " + name);
        }
    }

    tree_->expandAll();
    tree_->resizeColumnToContents(0);
    summary_->setText(QString("%1 prims  |  %2 meshes  |  %3 lights  |  %4 materials  |  %5 triangles  |  %6 polygons")
                          .arg(primCount)
                          .arg(stage ? stage->countOfType(PrimType::Mesh) : 0)
                          .arg(stage ? stage->countOfType(PrimType::Light) : 0)
                          .arg(materialContainers.size())
                          .arg(triangles)
                          .arg(polygons));

    if (!keepPath.isEmpty()) {
        if (QTreeWidgetItem* item = findItemByPath(tree_, keepPath)) {
            tree_->setCurrentItem(item);
            item->setSelected(true);
        }
    }
}

}  // namespace sol
