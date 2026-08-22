// Interactive render view: progressive framebuffer + Houdini-style tumble/pan/dolly
// and Translate / Rotate / Scale gizmos for selected transformable nodes.
#pragma once

#include <QImage>
#include <QString>
#include <QWidget>
#include <functional>

#include "core/math.h"
#include "nodes/node.h"

#include <QColor>

class QToolButton;
class QComboBox;
class QAction;

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
    // Idle / safe-start: cover-cropped bundled placeholder instead of a black FB.
    void showPlaceholder(bool show);
    bool isShowingPlaceholder() const { return showPlaceholder_; }
    // Crossfade placeholder → beauty over `durationMs` (samples may accumulate under).
    void beginPlaceholderFade(int durationMs = 1000);
    bool placeholderFadeActive() const { return fadeActive_; }
    void setStatusText(const QString& text) {
        statusText_ = text;
        update();
    }
    // Right side of the same bottom strip (e.g. dicing progress next to spp).
    void setStatusTextRight(const QString& text) {
        statusTextRight_ = text;
        update();
    }
    // Far-right HUD: live device (Embree / OptiX / XPU). Support text is empty on Embree.
    void setBackendHud(const QString& activeBackend, const QString& supportText,
                       const QColor& supportColor) {
        backendActive_ = activeBackend;
        optixSupportText_ = supportText;
        optixSupportColor_ = supportColor;
        hasBackendHud_ = true;
        update();
    }
    void setResolution(int width, int height);

    ViewCamera& camera() { return camera_; }
    const ViewCamera& camera() const { return camera_; }
    void setCamera(const ViewCamera& camera);
    bool navigationEnabled() const { return navigationEnabled_; }
    void setNavigationEnabled(bool enabled) { navigationEnabled_ = enabled; }
    bool isNavigating() const { return mode_ == 1 || mode_ == 2 || mode_ == 3; }

    TransformTool transformTool() const { return transformTool_; }
    void setTransformTool(TransformTool tool);
    TransformSpace transformSpace() const { return transformSpace_; }
    void setTransformSpace(TransformSpace space);
    int viewTransform() const { return viewTransform_; }
    void setViewTransform(int view);
    int colorManagement() const { return colorManagement_; }
    void setColorManagement(int mode);
    void setTransformTarget(Node* node);
    Node* transformTarget() const { return transformTarget_; }
    bool isGizmoDragging() const { return mode_ == 4; }

    // One-shot: next LMB on geometry sets camera DOF focus distance.
    bool focusPickActive() const { return focusPickActive_; }
    void setFocusPickActive(bool active);

    using PickCallback = std::function<bool(float u, float v, Vec3& hitPoint)>;
    void setPickCallback(PickCallback callback) { pickCallback_ = std::move(callback); }

    // Object pick for viewport selection: returns authoring node name for the hit.
    using ObjectPickCallback = std::function<bool(float u, float v, QString& sourceNode)>;
    void setObjectPickCallback(ObjectPickCallback callback) { objectPickCallback_ = std::move(callback); }

    // Returns world bounds for the current selection (false → frame whole scene).
    using SelectionBoundsCallback = std::function<bool(Bounds3& outBounds)>;
    void setSelectionBoundsCallback(SelectionBoundsCallback callback) {
        selectionBoundsCallback_ = std::move(callback);
    }
    using SceneBoundsCallback = std::function<bool(Bounds3& outBounds)>;
    void setSceneBoundsCallback(SceneBoundsCallback callback) { sceneBoundsCallback_ = std::move(callback); }

    // Houdini-style framing: F frames selection (or scene), H frames all.
    void frameBounds(const Bounds3& bounds);
    void frameSelection();
    void frameAll();

    // Look-through camera menu (Houdini-style). empty activeName → free view.
    void setCameraMenu(const QStringList& cameraNames, const QString& activeName);

    // Start / Stop live on the viewport chrome (left of camera / transform tools).
    void attachRenderActions(QAction* start, QAction* stop);

signals:
    void cameraMoved();
    // Fired while dragging (values already written quietly — do not cook/IPR).
    void transformEdited(sol::Node* node);
    // Fired on mouse release after a gizmo drag — safe to cook/IPR.
    void transformFinished(sol::Node* node);
    void transformToolChanged(sol::TransformTool tool);
    void transformSpaceChanged(sol::TransformSpace space);
    void viewTransformChanged(int view);
    void colorManagementChanged(int mode);
    // Viewport object pick: empty string clears selection.
    void objectSelected(const QString& sourceNode);
    // Focus-distance pick for DOF (metres along the view ray to the hit).
    void focusDistancePicked(float distanceMetres);
    void focusPickChanged(bool active);
    // Empty string = free perspective (no look-through camera).
    void lookThroughCameraChosen(const QString& cameraName);

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private:
    enum class GizmoAxis { None = 0, X, Y, Z, Center };

    QRect imageRect() const;
    QImage coverCroppedPlaceholder(const QSize& targetSize) const;
    bool pickUnderMouse(const QPoint& pos, Vec3& hitPoint) const;
    bool pickObjectUnderMouse(const QPoint& pos, QString& sourceNode) const;
    bool projectWorldToWidget(const Vec3& world, QPointF& out) const;
    bool widgetToCameraRay(const QPoint& pos, Vec3& origin, Vec3& direction) const;
    void beginNavigation(int mode, const QPoint& pos);
    void layoutToolStrip();
    void syncToolButtons();
    void rebuildCameraMenu();

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
    QImage placeholderImage_;
    bool showPlaceholder_ = false;
    bool fadeActive_ = false;
    qint64 fadeStartMs_ = 0;
    int fadeDurationMs_ = 1000;
    QString statusText_;
    QString statusTextRight_;
    QString backendActive_;
    QString optixSupportText_;
    QColor optixSupportColor_{255, 80, 80};
    bool hasBackendHud_ = false;
    ViewCamera camera_;
    PickCallback pickCallback_;
    ObjectPickCallback objectPickCallback_;
    SelectionBoundsCallback selectionBoundsCallback_;
    SceneBoundsCallback sceneBoundsCallback_;
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

    QWidget* chromeBar_ = nullptr;
    QWidget* renderControlStrip_ = nullptr;
    QWidget* toolStrip_ = nullptr;
    QToolButton* startButton_ = nullptr;
    QToolButton* stopButton_ = nullptr;
    QToolButton* cameraMenuButton_ = nullptr;
    QToolButton* homeButton_ = nullptr;
    QToolButton* selectButton_ = nullptr;
    QToolButton* translateButton_ = nullptr;
    QToolButton* rotateButton_ = nullptr;
    QToolButton* scaleButton_ = nullptr;
    QToolButton* localSpaceButton_ = nullptr;
    QToolButton* worldSpaceButton_ = nullptr;
    QComboBox* colorManagementCombo_ = nullptr;
    QComboBox* viewTransformCombo_ = nullptr;
    int colorManagement_ = 1;  // kColorOcio
    int viewTransform_ = 0;    // kViewSrgbAces
    QStringList cameraMenuNames_;
    QString activeCameraName_;

    // World-space tumble center for the current Alt+LMB / RMB drag (may differ from look-at).
    Vec3 tumbleCenter_{0.0f, 1.0f, 0.0f};
    bool showPivotMarker_ = false;
    Vec3 pivotMarkerWorld_{0.0f};
    qint64 pivotMarkerUntilMs_ = 0;
    bool focusPickActive_ = false;
};

}  // namespace sol
