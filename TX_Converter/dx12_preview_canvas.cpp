// Omega DX12 presenter: CPU-bakes 8-bit display (OCIO/grade), then streams to a HWND swapchain.
#include "dx12_preview_canvas.h"

#include <QPaintEngine>
#include <QPaintEvent>
#include <QResizeEvent>
#include <QShowEvent>

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef NOMINMAX
#  define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <wrl/client.h>

#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

namespace sol {
namespace {

constexpr UINT kFrameCount = 2;
constexpr UINT kTextureAlign = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;

const char* kShaderSrc = R"(
cbuffer Constants : register(b0) {
    float4 destRect;
    float2 viewSize;
    float  mode;
    float  _pad;
    float4 solidColor;
};

Texture2D    g_tex : register(t0);
SamplerState g_samp : register(s0);

struct VSOut {
    float4 pos : SV_Position;
};

VSOut VSMain(uint vid : SV_VertexID) {
    float2 uv = float2((vid << 1) & 2u, vid & 2u);
    VSOut o;
    o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

float4 PSMain(VSOut i) : SV_Target {
    if (mode < 0.5) return solidColor;
    float2 pixel = i.pos.xy;
    if (pixel.x < destRect.x || pixel.y < destRect.y ||
        pixel.x >= destRect.x + destRect.z || pixel.y >= destRect.y + destRect.w) {
        return solidColor;
    }
    float2 uv = (pixel - destRect.xy) / max(destRect.zw, float2(1e-3, 1e-3));
    return g_tex.SampleLevel(g_samp, uv, 0);
}
)";

struct CBData {
    float destRect[4];
    float viewSize[2];
    float mode;
    float pad;
    float solidColor[4];
};

}  // namespace

struct Dx12PreviewCanvas::Gpu {
    ComPtr<IDXGIFactory4> factory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain3> swapchain;
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12DescriptorHeap> srvHeap;
    ComPtr<ID3D12Resource> renderTargets[kFrameCount];
    ComPtr<ID3D12CommandAllocator> cmdAlloc;
    ComPtr<ID3D12GraphicsCommandList> cmdList;
    ComPtr<ID3D12RootSignature> rootSig;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12Resource> tex;
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Resource> cbUpload;
    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    UINT64 fenceValue = 0;
    UINT rtvDescriptorSize = 0;
    UINT frameIndex = 0;
    int swapW = 0;
    int swapH = 0;
    int texW = 0;
    int texH = 0;
    bool texCopyDest = false;
    bool allowTearing = false;
    bool rtInPresentState[kFrameCount] = {};
    HWND hwnd = nullptr;

    ~Gpu() {
        wait();
        if (fenceEvent) {
            CloseHandle(fenceEvent);
            fenceEvent = nullptr;
        }
    }

    void wait() {
        if (!queue || !fence || !fenceEvent) return;
        const UINT64 v = ++fenceValue;
        if (FAILED(queue->Signal(fence.Get(), v))) return;
        if (fence->GetCompletedValue() < v) {
            fence->SetEventOnCompletion(v, fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
    }
};

Dx12PreviewCanvas::Dx12PreviewCanvas(QWidget* parent) : FloatPreviewCanvas(parent) {
    setAttribute(Qt::WA_NativeWindow, true);
    setAttribute(Qt::WA_PaintOnScreen, true);
    setAttribute(Qt::WA_NoSystemBackground, true);
    setAttribute(Qt::WA_OpaquePaintEvent, true);
    setStyleSheet(QString());
}

Dx12PreviewCanvas::~Dx12PreviewCanvas() { releaseGpu(); }

QPaintEngine* Dx12PreviewCanvas::paintEngine() const { return nullptr; }

void Dx12PreviewCanvas::releaseGpu() {
    if (gpu_) {
        gpu_->wait();
        if (gpu_->swapchain) gpu_->swapchain->SetFullscreenState(FALSE, nullptr);
    }
    gpu_.reset();
}

bool Dx12PreviewCanvas::ensureGpu() {
    HWND hwnd = reinterpret_cast<HWND>(winId());
    if (!hwnd) return false;
    if (gpu_ && gpu_->hwnd == hwnd && gpu_->device) return true;

    releaseGpu();
    gpu_ = std::make_unique<Gpu>();
    gpu_->hwnd = hwnd;

    UINT factoryFlags = 0;
    if (FAILED(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&gpu_->factory)))) {
        releaseGpu();
        return false;
    }

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; gpu_->factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                        IID_PPV_ARGS(&gpu_->device)))) {
            break;
        }
        adapter.Reset();
        gpu_->device.Reset();
    }
    if (!gpu_->device) {
        ComPtr<IDXGIAdapter> warp;
        if (FAILED(gpu_->factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))) ||
            FAILED(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0,
                                     IID_PPV_ARGS(&gpu_->device)))) {
            releaseGpu();
            return false;
        }
    }

    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(gpu_->device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&gpu_->queue)))) {
        releaseGpu();
        return false;
    }

    BOOL allowTearing = FALSE;
    ComPtr<IDXGIFactory5> factory5;
    if (SUCCEEDED(gpu_->factory.As(&factory5)) &&
        SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing,
                                                 sizeof(allowTearing)))) {
        gpu_->allowTearing = allowTearing == TRUE;
    }

    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
    rtvHeapDesc.NumDescriptors = kFrameCount;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    if (FAILED(gpu_->device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&gpu_->rtvHeap)))) {
        releaseGpu();
        return false;
    }
    gpu_->rtvDescriptorSize =
        gpu_->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc{};
    srvHeapDesc.NumDescriptors = 1;
    srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(gpu_->device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&gpu_->srvHeap)))) {
        releaseGpu();
        return false;
    }

    if (FAILED(gpu_->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                     IID_PPV_ARGS(&gpu_->cmdAlloc))) ||
        FAILED(gpu_->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                gpu_->cmdAlloc.Get(), nullptr,
                                                IID_PPV_ARGS(&gpu_->cmdList)))) {
        releaseGpu();
        return false;
    }
    gpu_->cmdList->Close();

    D3D12_DESCRIPTOR_RANGE srvRange{};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = 1;
    srvRange.BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[0].Descriptor.ShaderRegister = 0;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &srvRange;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC samp{};
    samp.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samp.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samp.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samp.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samp.ShaderRegister = 0;
    samp.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    samp.MaxLOD = D3D12_FLOAT32_MAX;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = params;
    rsDesc.NumStaticSamplers = 1;
    rsDesc.pStaticSamplers = &samp;

    ComPtr<ID3DBlob> sigBlob;
    ComPtr<ID3DBlob> errBlob;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob,
                                           &errBlob)) ||
        FAILED(gpu_->device->CreateRootSignature(0, sigBlob->GetBufferPointer(),
                                                  sigBlob->GetBufferSize(),
                                                  IID_PPV_ARGS(&gpu_->rootSig)))) {
        releaseGpu();
        return false;
    }

    ComPtr<ID3DBlob> vs;
    ComPtr<ID3DBlob> ps;
    if (FAILED(D3DCompile(kShaderSrc, std::strlen(kShaderSrc), "preview", nullptr, nullptr,
                          "VSMain", "vs_5_0", 0, 0, &vs, &errBlob)) ||
        FAILED(D3DCompile(kShaderSrc, std::strlen(kShaderSrc), "preview", nullptr, nullptr,
                          "PSMain", "ps_5_0", 0, 0, &ps, &errBlob))) {
        releaseGpu();
        return false;
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = gpu_->rootSig.Get();
    psoDesc.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
    psoDesc.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.RasterizerState.DepthClipEnable = TRUE;
    psoDesc.DepthStencilState.DepthEnable = FALSE;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = 1;
    if (FAILED(gpu_->device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&gpu_->pso)))) {
        releaseGpu();
        return false;
    }

    D3D12_HEAP_PROPERTIES uploadHeap{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC cbDesc{};
    cbDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    cbDesc.Width = 256;
    cbDesc.Height = 1;
    cbDesc.DepthOrArraySize = 1;
    cbDesc.MipLevels = 1;
    cbDesc.SampleDesc.Count = 1;
    cbDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(gpu_->device->CreateCommittedResource(
            &uploadHeap, D3D12_HEAP_FLAG_NONE, &cbDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&gpu_->cbUpload)))) {
        releaseGpu();
        return false;
    }

    if (FAILED(gpu_->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gpu_->fence)))) {
        releaseGpu();
        return false;
    }
    gpu_->fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!gpu_->fenceEvent) {
        releaseGpu();
        return false;
    }

    // 1×1 placeholder so solid draws always have a valid SRV bound.
    {
        D3D12_RESOURCE_DESC texDesc{};
        texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDesc.Width = 1;
        texDesc.Height = 1;
        texDesc.DepthOrArraySize = 1;
        texDesc.MipLevels = 1;
        texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texDesc.SampleDesc.Count = 1;
        D3D12_HEAP_PROPERTIES defaultHeap{D3D12_HEAP_TYPE_DEFAULT};
        if (FAILED(gpu_->device->CreateCommittedResource(
                &defaultHeap, D3D12_HEAP_FLAG_NONE, &texDesc, D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr, IID_PPV_ARGS(&gpu_->tex)))) {
            releaseGpu();
            return false;
        }
        gpu_->texW = 1;
        gpu_->texH = 1;
        gpu_->texCopyDest = true;

        D3D12_RESOURCE_DESC upDesc{};
        upDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        upDesc.Width = kTextureAlign;
        upDesc.Height = 1;
        upDesc.DepthOrArraySize = 1;
        upDesc.MipLevels = 1;
        upDesc.SampleDesc.Count = 1;
        upDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES uploadHeap{D3D12_HEAP_TYPE_UPLOAD};
        if (FAILED(gpu_->device->CreateCommittedResource(
                &uploadHeap, D3D12_HEAP_FLAG_NONE, &upDesc, D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr, IID_PPV_ARGS(&gpu_->upload)))) {
            releaseGpu();
            return false;
        }
        void* mapped = nullptr;
        if (SUCCEEDED(gpu_->upload->Map(0, nullptr, &mapped))) {
            auto* p = static_cast<unsigned char*>(mapped);
            p[0] = p[1] = p[2] = 18;
            p[3] = 255;
            gpu_->upload->Unmap(0, nullptr);
        }
        gpu_->cmdAlloc->Reset();
        gpu_->cmdList->Reset(gpu_->cmdAlloc.Get(), nullptr);
        D3D12_TEXTURE_COPY_LOCATION dstLoc{};
        dstLoc.pResource = gpu_->tex.Get();
        dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION srcLoc{};
        srcLoc.pResource = gpu_->upload.Get();
        srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        srcLoc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srcLoc.PlacedFootprint.Footprint.Width = 1;
        srcLoc.PlacedFootprint.Footprint.Height = 1;
        srcLoc.PlacedFootprint.Footprint.Depth = 1;
        srcLoc.PlacedFootprint.Footprint.RowPitch = kTextureAlign;
        gpu_->cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        D3D12_RESOURCE_BARRIER toSrv{};
        toSrv.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        toSrv.Transition.pResource = gpu_->tex.Get();
        toSrv.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        toSrv.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        toSrv.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        gpu_->cmdList->ResourceBarrier(1, &toSrv);
        gpu_->cmdList->Close();
        ID3D12CommandList* lists[] = {gpu_->cmdList.Get()};
        gpu_->queue->ExecuteCommandLists(1, lists);
        gpu_->wait();
        gpu_->texCopyDest = false;

        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        gpu_->device->CreateShaderResourceView(gpu_->tex.Get(), &srv,
                                               gpu_->srvHeap->GetCPUDescriptorHandleForHeapStart());
    }
    return true;
}

bool Dx12PreviewCanvas::ensureSwapchain(int physicalW, int physicalH) {
    if (!gpu_ || !gpu_->device) return false;
    int w = std::max(1, physicalW);
    int h = std::max(1, physicalH);
    if (gpu_->swapchain && gpu_->swapW == w && gpu_->swapH == h) return true;

    gpu_->wait();
    for (UINT i = 0; i < kFrameCount; ++i) gpu_->renderTargets[i].Reset();

    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = UINT(w);
    desc.Height = UINT(h);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = kFrameCount;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.Flags = gpu_->allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

    if (!gpu_->swapchain) {
        ComPtr<IDXGISwapChain1> sc1;
        if (FAILED(gpu_->factory->CreateSwapChainForHwnd(gpu_->queue.Get(), gpu_->hwnd, &desc,
                                                          nullptr, nullptr, &sc1))) {
            return false;
        }
        gpu_->factory->MakeWindowAssociation(gpu_->hwnd, DXGI_MWA_NO_ALT_ENTER);
        if (FAILED(sc1.As(&gpu_->swapchain))) return false;
    } else if (FAILED(gpu_->swapchain->ResizeBuffers(kFrameCount, desc.Width, desc.Height,
                                                      desc.Format, desc.Flags))) {
        return false;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE rtv = gpu_->rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < kFrameCount; ++i) {
        if (FAILED(gpu_->swapchain->GetBuffer(i, IID_PPV_ARGS(&gpu_->renderTargets[i])))) {
            return false;
        }
        gpu_->device->CreateRenderTargetView(gpu_->renderTargets[i].Get(), nullptr, rtv);
        rtv.ptr += gpu_->rtvDescriptorSize;
    }
    gpu_->frameIndex = gpu_->swapchain->GetCurrentBackBufferIndex();
    gpu_->swapW = w;
    gpu_->swapH = h;
    for (UINT i = 0; i < kFrameCount; ++i) gpu_->rtInPresentState[i] = false;
    return true;
}

void Dx12PreviewCanvas::showEvent(QShowEvent* event) {
    FloatPreviewCanvas::showEvent(event);
    ensureGpu();
    update();
}

void Dx12PreviewCanvas::resizeEvent(QResizeEvent* event) {
    FloatPreviewCanvas::resizeEvent(event);
    if (gpu_ && gpu_->device) {
        const qreal dpr = devicePixelRatioF();
        ensureSwapchain(int(std::lround(width() * dpr)), int(std::lround(height() * dpr)));
    }
}

void Dx12PreviewCanvas::paintEvent(QPaintEvent*) {
    if (!hasLinearImage()) {
        presentSolid(18.0f / 255.0f, 20.0f / 255.0f, 22.0f / 255.0f);
        return;
    }
    ensureDisplayCache();
    if (displayCacheImage().isNull()) {
        presentSolid(18.0f / 255.0f, 20.0f / 255.0f, 22.0f / 255.0f);
        return;
    }
    presentImage(displayCacheImage(), imageRect());
}

void Dx12PreviewCanvas::presentSolid(float r, float g, float b) {
    if (!ensureGpu()) return;
    const qreal dpr = devicePixelRatioF();
    const int pw = std::max(1, int(std::lround(width() * dpr)));
    const int ph = std::max(1, int(std::lround(height() * dpr)));
    if (!ensureSwapchain(pw, ph)) return;

    CBData cb{};
    cb.destRect[0] = 0;
    cb.destRect[1] = 0;
    cb.destRect[2] = float(pw);
    cb.destRect[3] = float(ph);
    cb.viewSize[0] = float(pw);
    cb.viewSize[1] = float(ph);
    cb.mode = 0.0f;
    cb.solidColor[0] = r;
    cb.solidColor[1] = g;
    cb.solidColor[2] = b;
    cb.solidColor[3] = 1.0f;

    void* mapped = nullptr;
    if (FAILED(gpu_->cbUpload->Map(0, nullptr, &mapped))) return;
    std::memcpy(mapped, &cb, sizeof(cb));
    gpu_->cbUpload->Unmap(0, nullptr);

    gpu_->wait();
    gpu_->cmdAlloc->Reset();
    gpu_->cmdList->Reset(gpu_->cmdAlloc.Get(), gpu_->pso.Get());

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = gpu_->renderTargets[gpu_->frameIndex].Get();
    barrier.Transition.StateBefore = gpu_->rtInPresentState[gpu_->frameIndex]
                                         ? D3D12_RESOURCE_STATE_PRESENT
                                         : D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    gpu_->cmdList->ResourceBarrier(1, &barrier);

    D3D12_CPU_DESCRIPTOR_HANDLE rtv = gpu_->rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += size_t(gpu_->frameIndex) * gpu_->rtvDescriptorSize;
    gpu_->cmdList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

    const D3D12_VIEWPORT vp{0.0f, 0.0f, float(pw), float(ph), 0.0f, 1.0f};
    const D3D12_RECT sc{0, 0, LONG(pw), LONG(ph)};
    gpu_->cmdList->RSSetViewports(1, &vp);
    gpu_->cmdList->RSSetScissorRects(1, &sc);
    gpu_->cmdList->SetGraphicsRootSignature(gpu_->rootSig.Get());
    gpu_->cmdList->SetGraphicsRootConstantBufferView(0, gpu_->cbUpload->GetGPUVirtualAddress());
    ID3D12DescriptorHeap* heaps[] = {gpu_->srvHeap.Get()};
    gpu_->cmdList->SetDescriptorHeaps(1, heaps);
    gpu_->cmdList->SetGraphicsRootDescriptorTable(
        1, gpu_->srvHeap->GetGPUDescriptorHandleForHeapStart());
    gpu_->cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    gpu_->cmdList->DrawInstanced(3, 1, 0, 0);

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    gpu_->cmdList->ResourceBarrier(1, &barrier);
    gpu_->cmdList->Close();

    ID3D12CommandList* lists[] = {gpu_->cmdList.Get()};
    gpu_->queue->ExecuteCommandLists(1, lists);

    UINT presentFlags = 0;
    if (gpu_->allowTearing) presentFlags |= DXGI_PRESENT_ALLOW_TEARING;
    gpu_->swapchain->Present(0, presentFlags);
    gpu_->rtInPresentState[gpu_->frameIndex] = true;
    gpu_->frameIndex = gpu_->swapchain->GetCurrentBackBufferIndex();
    gpu_->wait();
}

void Dx12PreviewCanvas::presentImage(const QImage& rgb888, const QRectF& destLogical) {
    if (!ensureGpu()) return;
    if (rgb888.isNull() || rgb888.width() < 1 || rgb888.height() < 1) {
        presentSolid(18.0f / 255.0f, 20.0f / 255.0f, 22.0f / 255.0f);
        return;
    }

    const qreal dpr = devicePixelRatioF();
    const int pw = std::max(1, int(std::lround(width() * dpr)));
    const int ph = std::max(1, int(std::lround(height() * dpr)));
    if (!ensureSwapchain(pw, ph)) return;

    QImage src = (rgb888.format() == QImage::Format_RGB888)
                     ? rgb888
                     : rgb888.convertToFormat(QImage::Format_RGB888);
    const int tw = src.width();
    const int th = src.height();

    if (!gpu_->tex || gpu_->texW != tw || gpu_->texH != th) {
        gpu_->wait();
        gpu_->tex.Reset();
        gpu_->upload.Reset();
        gpu_->texW = tw;
        gpu_->texH = th;

        D3D12_RESOURCE_DESC texDesc{};
        texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDesc.Width = UINT(tw);
        texDesc.Height = UINT(th);
        texDesc.DepthOrArraySize = 1;
        texDesc.MipLevels = 1;
        texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texDesc.SampleDesc.Count = 1;

        D3D12_HEAP_PROPERTIES defaultHeap{D3D12_HEAP_TYPE_DEFAULT};
        if (FAILED(gpu_->device->CreateCommittedResource(
                &defaultHeap, D3D12_HEAP_FLAG_NONE, &texDesc, D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr, IID_PPV_ARGS(&gpu_->tex)))) {
            return;
        }
        gpu_->texCopyDest = true;

        const UINT64 rowPitch =
            (UINT64(tw) * 4 + (kTextureAlign - 1)) & ~(UINT64(kTextureAlign) - 1);
        D3D12_RESOURCE_DESC upDesc{};
        upDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        upDesc.Width = rowPitch * UINT64(th);
        upDesc.Height = 1;
        upDesc.DepthOrArraySize = 1;
        upDesc.MipLevels = 1;
        upDesc.SampleDesc.Count = 1;
        upDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES uploadHeap{D3D12_HEAP_TYPE_UPLOAD};
        if (FAILED(gpu_->device->CreateCommittedResource(
                &uploadHeap, D3D12_HEAP_FLAG_NONE, &upDesc, D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr, IID_PPV_ARGS(&gpu_->upload)))) {
            gpu_->tex.Reset();
            return;
        }

        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        gpu_->device->CreateShaderResourceView(gpu_->tex.Get(), &srv,
                                               gpu_->srvHeap->GetCPUDescriptorHandleForHeapStart());
    }

    {
        const UINT64 rowPitch =
            (UINT64(tw) * 4 + (kTextureAlign - 1)) & ~(UINT64(kTextureAlign) - 1);
        void* mapped = nullptr;
        D3D12_RANGE readRange{0, 0};
        if (FAILED(gpu_->upload->Map(0, &readRange, &mapped))) return;
        auto* dstBase = static_cast<unsigned char*>(mapped);
        for (int y = 0; y < th; ++y) {
            const uchar* line = src.constScanLine(y);
            unsigned char* dst = dstBase + size_t(y) * size_t(rowPitch);
            for (int x = 0; x < tw; ++x) {
                dst[x * 4 + 0] = line[x * 3 + 0];
                dst[x * 4 + 1] = line[x * 3 + 1];
                dst[x * 4 + 2] = line[x * 3 + 2];
                dst[x * 4 + 3] = 255;
            }
        }
        gpu_->upload->Unmap(0, nullptr);
    }

    CBData cb{};
    cb.destRect[0] = float(destLogical.x() * dpr);
    cb.destRect[1] = float(destLogical.y() * dpr);
    cb.destRect[2] = float(std::max(1.0, destLogical.width() * dpr));
    cb.destRect[3] = float(std::max(1.0, destLogical.height() * dpr));
    cb.viewSize[0] = float(pw);
    cb.viewSize[1] = float(ph);
    cb.mode = 1.0f;
    cb.solidColor[0] = 18.0f / 255.0f;
    cb.solidColor[1] = 20.0f / 255.0f;
    cb.solidColor[2] = 22.0f / 255.0f;
    cb.solidColor[3] = 1.0f;
    {
        void* mapped = nullptr;
        if (FAILED(gpu_->cbUpload->Map(0, nullptr, &mapped))) return;
        std::memcpy(mapped, &cb, sizeof(cb));
        gpu_->cbUpload->Unmap(0, nullptr);
    }

    gpu_->wait();
    gpu_->cmdAlloc->Reset();
    gpu_->cmdList->Reset(gpu_->cmdAlloc.Get(), gpu_->pso.Get());

    if (!gpu_->texCopyDest) {
        D3D12_RESOURCE_BARRIER toCopy{};
        toCopy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        toCopy.Transition.pResource = gpu_->tex.Get();
        toCopy.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        toCopy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        toCopy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        gpu_->cmdList->ResourceBarrier(1, &toCopy);
    }

    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = gpu_->tex.Get();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = gpu_->upload.Get();
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint.Offset = 0;
    srcLoc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srcLoc.PlacedFootprint.Footprint.Width = UINT(tw);
    srcLoc.PlacedFootprint.Footprint.Height = UINT(th);
    srcLoc.PlacedFootprint.Footprint.Depth = 1;
    srcLoc.PlacedFootprint.Footprint.RowPitch = UINT(
        (UINT64(tw) * 4 + (kTextureAlign - 1)) & ~(UINT64(kTextureAlign) - 1));

    gpu_->cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

    {
        D3D12_RESOURCE_BARRIER toSrv{};
        toSrv.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        toSrv.Transition.pResource = gpu_->tex.Get();
        toSrv.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        toSrv.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        toSrv.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        gpu_->cmdList->ResourceBarrier(1, &toSrv);
        gpu_->texCopyDest = false;
    }

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = gpu_->renderTargets[gpu_->frameIndex].Get();
    barrier.Transition.StateBefore = gpu_->rtInPresentState[gpu_->frameIndex]
                                         ? D3D12_RESOURCE_STATE_PRESENT
                                         : D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    gpu_->cmdList->ResourceBarrier(1, &barrier);

    D3D12_CPU_DESCRIPTOR_HANDLE rtv = gpu_->rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += size_t(gpu_->frameIndex) * gpu_->rtvDescriptorSize;
    gpu_->cmdList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

    const D3D12_VIEWPORT vp{0.0f, 0.0f, float(pw), float(ph), 0.0f, 1.0f};
    const D3D12_RECT sc{0, 0, LONG(pw), LONG(ph)};
    gpu_->cmdList->RSSetViewports(1, &vp);
    gpu_->cmdList->RSSetScissorRects(1, &sc);
    gpu_->cmdList->SetGraphicsRootSignature(gpu_->rootSig.Get());
    gpu_->cmdList->SetGraphicsRootConstantBufferView(0, gpu_->cbUpload->GetGPUVirtualAddress());
    ID3D12DescriptorHeap* heaps[] = {gpu_->srvHeap.Get()};
    gpu_->cmdList->SetDescriptorHeaps(1, heaps);
    gpu_->cmdList->SetGraphicsRootDescriptorTable(
        1, gpu_->srvHeap->GetGPUDescriptorHandleForHeapStart());
    gpu_->cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    gpu_->cmdList->DrawInstanced(3, 1, 0, 0);

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    gpu_->cmdList->ResourceBarrier(1, &barrier);
    gpu_->cmdList->Close();

    ID3D12CommandList* lists[] = {gpu_->cmdList.Get()};
    gpu_->queue->ExecuteCommandLists(1, lists);

    UINT presentFlags = 0;
    if (gpu_->allowTearing) presentFlags |= DXGI_PRESENT_ALLOW_TEARING;
    gpu_->swapchain->Present(0, presentFlags);
    gpu_->rtInPresentState[gpu_->frameIndex] = true;
    gpu_->frameIndex = gpu_->swapchain->GetCurrentBackBufferIndex();
    gpu_->wait();
}

}  // namespace sol
