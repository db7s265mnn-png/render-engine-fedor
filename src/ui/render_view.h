// Interactive render view: progressive framebuffer + Houdini-style tumble/pan/dolly
// and Translate / Rotate / Scale gizmos for selected transformable nodes.
#pragma once

#include <QImage>
#include <QWidget>
#include <functional>

#include "core/math.h"
#include "nodes/node.h"

class QToolButton;

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

enum class TransformTool { Select = 0, Translate, Rotate, Scale };
enum class TransformSpace { Local = 0, World };

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

    TransformTool transformTool() const { return transformTool_; }
    void setTransformTool(TransformTool tool);
    TransformSpace transformSpace() const { return transformSpace_; }
    void setTransformSpace(TransformSpace space);
    void setTransformTarget(Node* node);
    Node* transformTarget() const { return transformTarget_; }
    bool isGizmoDragging() const { return mode_ == 4; }

    using PickCallback = std::function<bool(float u, float v, Vec3& hitPoint)>;
    void setPickCallback(PickCallback callback) { pickCallback_ = std::move(callback); }

signals:
    void cameraMoved();
    // Fired while dragging (values already written quietly — do not cook/IPR).
    void transformEdited(sol::Node* node);
    // Fired on mouse release after a gizmo drag — safe to cook/IPR.
    void transformFinished(sol::Node* node);
    void transformToolChanged(sol::TransformTool tool);
    void transformSpaceChanged(sol::TransformSpace space);

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    enum class GizmoAxis { None = 0, X, Y, Z, Center };

    QRect imageRect() const;
    bool pickUnderMouse(const QPoint& pos, Vec3& hitPoint) const;
    bool projectWorldToWidget(const Vec3& world, QPointF& out) const;
    bool widgetToCameraRay(const QPoint& pos, Vec3& origin, Vec3& direction) const;
    void beginNavigation(int mode, const QPoint& pos);
    void layoutToolStrip();
    void syncToolButtons();

    bool hasTransformTarget() const;
    Mat4 targetWorldMatrix() const;
    Vec3 targetOrigin() const;
    void targetAxes(Vec3& x, Vec3& y, Vec3& z) const;
    float gizmoWorldSize() const;
    bool rayPlaneHit(const Vec3& rayO, const Vec3& rayD, const Vec3& planeO, const Vec3& planeN,
                     Vec3& out) const;
    float angleOnAxisPlane(const Vec3& axis, const Vec3& center, const Vec3& point) const;
    bool ringAngleAtMouse(const QPoint& pos, const Vec3& axis, float& angleOut) const;
    float ringScreenDistance(const QPoint& pos, const Vec3& axis, float radius) const;
    GizmoAxis hitTestGizmo(const QPoint& pos) const;
    bool beginGizmoDrag(const QPoint& pos);
    void updateGizmoDrag(const QPoint& pos);
    void endGizmoDrag();
    void drawGizmo(QPainter& painter);
    void drawRotationRings(QPainter& painter, const Vec3& origin, const Vec3& ax, const Vec3& ay,
                           const Vec3& az, float radius);

    QImage image_;
    QString statusText_;
    ViewCamera camera_;
    PickCallback pickCallback_;
    QPoint lastMousePosition_;
    int mode_ = 0;  // 0 none, 1 orbit, 2 pan, 3 dolly, 4 gizmo
    bool navigationEnabled_ = true;
    int resolutionX_ = 960;
    int resolutionY_ = 540;

    TransformTool transformTool_ = TransformTool::Translate;
    TransformSpace transformSpace_ = TransformSpace::Local;
    Node* transformTarget_ = nullptr;
    GizmoAxis activeAxis_ = GizmoAxis::None;
    GizmoAxis hoverAxis_ = GizmoAxis::None;
    Vec3 dragStartTranslate_{0.0f};
    Vec3 dragStartRotate_{0.0f};
    Vec3 dragStartScale_{1.0f};
    Mat4 dragStartMatrix_ = Mat4::identity();
    Vec3 dragAxisDir_{1.0f, 0.0f, 0.0f};
    Vec3 dragOrigin_{0.0f};
    float dragStartParam_ = 0.0f;
    float dragStartAngle_ = 0.0f;
    QPoint dragStartMouse_;
    QString dragParameterName_;
    bool gizmoDidEdit_ = false;

    QWidget* toolStrip_ = nullptr;
    QToolButton* translateButton_ = nullptr;
    QToolButton* rotateButton_ = nullptr;
    QToolButton* scaleButton_ = nullptr;
    QToolButton* localSpaceButton_ = nullptr;
    QToolButton* worldSpaceButton_ = nullptr;

    // World-space tumble center for the current Alt+LMB / RMB drag (may differ from look-at).
    Vec3 tumbleCenter_{0.0f, 1.0f, 0.0f};
    bool showPivotMarker_ = false;
    Vec3 pivotMarkerWorld_{0.0f};
    qint64 pivotMarkerUntilMs_ = 0;
};

}  // namespace sol
