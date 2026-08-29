// Scene undo/redo (QUndoStack, limit 100). Parameter sliders and camera
// navigation coalesce until mouse release (Blender-style).
#pragma once

#include <QHash>
#include <QJsonObject>
#include <QObject>
#include <QPointF>
#include <QString>
#include <QUndoStack>
#include <QVariant>
#include <cmath>
#include <functional>

#include "core/math.h"

namespace sol {

class NodeGraph;

struct OrbitCameraState {
    Vec3 pivot{0.0f, 1.0f, 0.0f};
    float distance = 10.0f;
    float yaw = 30.0f;
    float pitch = -20.0f;
    QString nodeName;
    QVariant eye;
    QVariant target;
    QVariant translate;
    bool hasNode = false;
    bool uselookat = true;
};

inline bool orbitCameraStateEqual(const OrbitCameraState& a, const OrbitCameraState& b) {
    auto close = [](float x, float y) { return std::fabs(x - y) < 1e-5f; };
    auto vclose = [&](const Vec3& x, const Vec3& y) {
        return close(x.x, y.x) && close(x.y, y.y) && close(x.z, y.z);
    };
    if (!vclose(a.pivot, b.pivot) || !close(a.distance, b.distance) || !close(a.yaw, b.yaw) ||
        !close(a.pitch, b.pitch))
        return false;
    if (a.hasNode != b.hasNode || a.nodeName != b.nodeName || a.uselookat != b.uselookat) return false;
    if (a.eye != b.eye || a.target != b.target || a.translate != b.translate) return false;
    return true;
}

class UndoHub : public QObject {
    Q_OBJECT

public:
    explicit UndoHub(QObject* parent = nullptr);

    QUndoStack& stack() { return stack_; }
    const QUndoStack& stack() const { return stack_; }

    void setGraph(NodeGraph* graph) { graph_ = graph; }
    void setRefreshUi(std::function<void()> fn) { refreshUi_ = std::move(fn); }
    void setRebuildGraph(std::function<void()> fn) { rebuildGraph_ = std::move(fn); }
    void setApplyCamera(std::function<void(const OrbitCameraState&)> fn) { applyCamera_ = std::move(fn); }

    bool ignoring() const { return ignore_ > 0; }

    class IgnoreGuard {
    public:
        explicit IgnoreGuard(UndoHub& hub) : hub_(&hub) { ++hub_->ignore_; }
        ~IgnoreGuard() {
            if (hub_) --hub_->ignore_;
        }
        IgnoreGuard(const IgnoreGuard&) = delete;
        IgnoreGuard& operator=(const IgnoreGuard&) = delete;

    private:
        UndoHub* hub_ = nullptr;
    };

    void pushParameter(const QString& nodeName, const QString& param, const QVariant& oldValue,
                       const QVariant& newValue, const QString& text = QString());
    void beginParameter(const QString& nodeName, const QString& param, const QVariant& oldValue);
    void endParameter(const QString& nodeName, const QString& param, const QVariant& newValue);

    void pushGraphSnapshot(const QJsonObject& before, const QJsonObject& after, const QString& text);
    void pushNodePositions(const QHash<QString, QPointF>& oldPos, const QHash<QString, QPointF>& newPos);

    void pushCamera(const OrbitCameraState& before, const OrbitCameraState& after);

    void beginMacro(const QString& text);
    void endMacro();

    void clear();
    void undo();
    void redo();

    NodeGraph* graph() const { return graph_; }
    void refreshUi() {
        if (refreshUi_) refreshUi_();
    }
    void rebuildGraph() {
        if (rebuildGraph_) rebuildGraph_();
    }
    void applyCamera(const OrbitCameraState& state) {
        if (applyCamera_) applyCamera_(state);
    }

private:
    QUndoStack stack_;
    NodeGraph* graph_ = nullptr;
    std::function<void()> refreshUi_;
    std::function<void()> rebuildGraph_;
    std::function<void(const OrbitCameraState&)> applyCamera_;
    int ignore_ = 0;

    QString dragNode_;
    QString dragParam_;
    QVariant dragOld_;
    bool dragging_ = false;
};

}  // namespace sol
