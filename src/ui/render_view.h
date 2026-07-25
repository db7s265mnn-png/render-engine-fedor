// Interactive render view: progressive framebuffer + Houdini-style tumble/pan/dolly.
#pragma once

#include <QImage>
#include <QWidget>
#include <functional>

#include "core/math.h"

namespace sol {

struct ViewCamera {
    Vec3 pivot{0.0f, 1.0f, 0.0f};
    float distance = 10.0f;
    float yaw = 30.0f;     // degrees around +Y
    float pitch = -20.0f;  // degrees, negative looks down

    Mat4 toMatrix() const;
    Vec3 eye() const;
    void setFromMatrix(const Mat4& cameraToWorld, float focusDistance);
    // Classic orbit around the look-at pivot.
    void orbit(float deltaYaw, float deltaPitch);
    // Rotate eye + look-at around a fixed world point without reframing on set.
    void orbitAround(const Vec3& center, float deltaYaw, float deltaPitch);
    void pan(float dx, float dy);
    void dolly(float amount);
};

class RenderView : public QWidget {
    Q_OBJECT

public:
    explicit RenderView(QWidget* parent = nullptr);

    void setImage(const QImage& image);
    void clearImage();
    void setStatusText(const QString& text) {
        statusText_ = text;
        update();
    }
    void setResolution(int width, int height);

    ViewCamera& camera() { return camera_; }
    const ViewCamera& camera() const { return camera_; }
    void setCamera(const ViewCamera& camera);
    bool navigationEnabled() const { return navigationEnabled_; }
    void setNavigationEnabled(bool enabled) { navigationEnabled_ = enabled; }

    using PickCallback = std::function<bool(float u, float v, Vec3& hitPoint)>;
    void setPickCallback(PickCallback callback) { pickCallback_ = std::move(callback); }

signals:
    void cameraMoved();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    QRect imageRect() const;
    bool pickUnderMouse(const QPoint& pos, Vec3& hitPoint) const;
    bool projectWorldToWidget(const Vec3& world, QPointF& out) const;
    void beginNavigation(int mode, const QPoint& pos);

    QImage image_;
    QString statusText_;
    ViewCamera camera_;
    PickCallback pickCallback_;
    QPoint lastMousePosition_;
    int mode_ = 0;  // 0 none, 1 orbit, 2 pan, 3 dolly
    bool navigationEnabled_ = true;
    int resolutionX_ = 960;
    int resolutionY_ = 540;

    // World-space tumble center for the current Alt+LMB / RMB drag (may differ from look-at).
    Vec3 tumbleCenter_{0.0f, 1.0f, 0.0f};
    bool showPivotMarker_ = false;
    Vec3 pivotMarkerWorld_{0.0f};
    qint64 pivotMarkerUntilMs_ = 0;
};

}  // namespace sol
