// Omega viewport: same CPU display bake as Alpha, then stream 8-bit RGB to DX12.
#pragma once

#include "texture_viewer.h"

#include <memory>

class QShowEvent;

namespace sol {

class Dx12PreviewCanvas : public FloatPreviewCanvas {
    Q_OBJECT
public:
    explicit Dx12PreviewCanvas(QWidget* parent = nullptr);
    ~Dx12PreviewCanvas() override;

protected:
    void showEvent(QShowEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    QPaintEngine* paintEngine() const override;

private:
    struct Gpu;
    std::unique_ptr<Gpu> gpu_;

    bool ensureGpu();
    bool ensureSwapchain(int physicalW, int physicalH);
    void releaseGpu();
    void presentSolid(float r, float g, float b);
    void presentImage(const QImage& rgb888, const QRectF& destLogical);
};

}  // namespace sol
