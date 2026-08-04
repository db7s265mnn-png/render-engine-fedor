// Standalone TX Converter — maketx/OCIO core + texture / UDIM preview.
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QIcon>
#include <QImageReader>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QSplitter>
#include <QSize>
#include <QEventLoop>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>

#include "core/log.h"
#include "core/expr_eval.h"
#include "io/image_io.h"
#include "io/ocio_util.h"
#include "io/tx_convert.h"
#include "texture_viewer.h"
#include "ui/texture_file_dialog.h"
#include "ui/theme.h"

namespace {

class TxConverterWindow : public QWidget {
public:
    TxConverterWindow() {
#ifndef SOLSTICE_TX_CONVERTER_NAME
#define SOLSTICE_TX_CONVERTER_NAME "Grendizer_TX_Converter"
#endif
        setWindowTitle(QStringLiteral(SOLSTICE_TX_CONVERTER_NAME));
        QIcon appIcon(QStringLiteral(":/icons/app_icon.png"));
        appIcon.addFile(QStringLiteral(":/icons/app_icon_32.png"), QSize(32, 32));
        appIcon.addFile(QStringLiteral(":/icons/app_icon_64.png"), QSize(64, 64));
        setWindowIcon(appIcon);
        resize(1280, 780);

        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(8, 8, 8, 8);
        root->setSpacing(6);

        auto* splitter = new QSplitter(Qt::Horizontal, this);
        splitter->setChildrenCollapsible(false);

        auto* leftPanel = new QWidget();
        leftPanel->setMinimumWidth(320);
        leftPanel->setMaximumWidth(520);
        auto* leftLay = new QVBoxLayout(leftPanel);
        leftLay->setContentsMargins(0, 0, 4, 0);
        leftLay->setSpacing(8);

        auto* formBox = new QGroupBox(QStringLiteral("Convert"));
        form_ = new QFormLayout(formBox);

        auto makeBrowseRow = [&](QLineEdit** editOut, bool directory) {
            auto* row = new QWidget();
            auto* lay = new QHBoxLayout(row);
            lay->setContentsMargins(0, 0, 0, 0);
            auto* edit = new QLineEdit();
            auto* browse = new QPushButton(QStringLiteral("…"));
            browse->setFixedWidth(32);
            lay->addWidget(edit, 1);
            lay->addWidget(browse);
            *editOut = edit;
            connect(browse, &QPushButton::clicked, this, [this, edit, directory] {
                QString path;
                if (directory) {
                    path = QFileDialog::getExistingDirectory(this, QStringLiteral("Choose folder"),
                                                            edit->text());
                } else {
                    const auto picked = sol::TextureFileDialog::getOpenTexture(
                        this, QStringLiteral("Choose texture"), edit->text(),
                        QStringLiteral(
                            "Images (*.png *.jpg *.jpeg *.exr *.hdr *.tif *.tiff *.bmp *.tx);;All (*)"));
                    path = picked.path;
                }
                if (!path.isEmpty()) {
                    edit->setText(path);
                    syncFormatUi();
                    syncPipeline();
                }
            });
            return row;
        };

        form_->addRow(QStringLiteral("Source"), makeBrowseRow(&sourceEdit_, false));
        sourceEdit_->setPlaceholderText(QStringLiteral("texture.png or tile_<UDIM>.exr / .tx"));
        form_->addRow(QStringLiteral("Output Folder"), makeBrowseRow(&outputEdit_, true));
        outputEdit_->setPlaceholderText(QStringLiteral("Choose an output folder…"));
        outputEdit_->clear();

        formatCombo_ = new QComboBox();
        formatCombo_->addItem(QStringLiteral("Original"), int(sol::TxOutputFormat::Original));
        formatCombo_->addItem(QStringLiteral("EXR"), int(sol::TxOutputFormat::Exr));
        formatCombo_->addItem(QStringLiteral("TIFF"), int(sol::TxOutputFormat::Tiff));
        formatCombo_->addItem(QStringLiteral("TX"), int(sol::TxOutputFormat::Tx));
        formatCombo_->addItem(QStringLiteral("PNG"), int(sol::TxOutputFormat::Png));
        formatCombo_->addItem(QStringLiteral("JPG"), int(sol::TxOutputFormat::Jpg));
        formatCombo_->setCurrentIndex(formatCombo_->findData(int(sol::TxOutputFormat::Tx)));
        form_->addRow(QStringLiteral("Format"), formatCombo_);

        bitDepthCombo_ = new QComboBox();
        form_->addRow(QStringLiteral("Bit Depth"), bitDepthCombo_);

        resolutionCombo_ = new QComboBox();
        resolutionCombo_->addItem(QStringLiteral("Original"), 0);
        for (int s : {256, 512, 1024, 2048, 4096, 8192})
            resolutionCombo_->addItem(QString::number(s), s);
        resolutionCombo_->setToolTip(
            QStringLiteral("Long side in pixels (aspect preserved). Original = no resize."));
        form_->addRow(QStringLiteral("Resolution"), resolutionCombo_);

        channelsCombo_ = new QComboBox();
        form_->addRow(QStringLiteral("Channels"), channelsCombo_);

        colorSpaceCombo_ = new QComboBox();
        colorSpaceCombo_->setEditable(true);
        refreshColorSpaces(false);
        form_->addRow(QStringLiteral("Color Space"), colorSpaceCombo_);
        colorSpaceLabel_ = form_->labelForField(colorSpaceCombo_);

        advancedCheck_ = new QCheckBox(QStringLiteral("Advanced (full OCIO list)"));
        form_->addRow(QString(), advancedCheck_);
        connect(advancedCheck_, &QCheckBox::toggled, this, [this](bool on) { refreshColorSpaces(on); });

        useEnvCheck_ = new QCheckBox(QStringLiteral("Use OCIO from Environment"));
        useEnvCheck_->setChecked(true);
        useEnvCheck_->setToolTip(QStringLiteral("Read config path from the OCIO environment variable."));
        form_->addRow(QString(), useEnvCheck_);

        ocioEdit_ = new QLineEdit();
        auto* ocioRow = new QWidget();
        auto* ocioLay = new QHBoxLayout(ocioRow);
        ocioLay->setContentsMargins(0, 0, 0, 0);
        auto* ocioBrowse = new QPushButton(QStringLiteral("…"));
        ocioBrowse->setFixedWidth(32);
        ocioLay->addWidget(ocioEdit_, 1);
        ocioLay->addWidget(ocioBrowse);
        form_->addRow(QStringLiteral("OCIO Config"), ocioRow);
        ocioLabel_ = form_->labelForField(ocioRow);
        connect(ocioBrowse, &QPushButton::clicked, this, [this] {
            const QString path = QFileDialog::getOpenFileName(
                this, QStringLiteral("OCIO config"), ocioEdit_->text(),
                QStringLiteral("OCIO (*.ocio);;All (*)"));
            if (!path.isEmpty()) ocioEdit_->setText(path);
        });
        auto syncOcioEnabled = [this] {
            const bool manual = !useEnvCheck_->isChecked();
            ocioEdit_->setEnabled(manual);
        };
        connect(useEnvCheck_, &QCheckBox::toggled, this, [this, syncOcioEnabled](bool) {
            syncOcioEnabled();
            syncViewerOcio();
            refreshOcioStatus();
        });
        connect(ocioEdit_, &QLineEdit::editingFinished, this, [this] {
            syncViewerOcio();
            refreshOcioStatus();
        });
        syncOcioEnabled();

        connect(formatCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                [this](int) {
                    syncFormatUi();
                    syncPipeline();
                });
        syncFormatUi();

        leftLay->addWidget(formBox);

        auto* hint = new QLabel(
            QStringLiteral("TX → ACEScg. EXR/TIFF/PNG/JPG/Original keep source colour "
                           "(resize / bit / channels only). "
                           "R/G/B/A write 1 channel; RGB = 3; RGBA = 4. "
                           "Output folder must be chosen. F = fit, 1/2 = Source/Output."));
        hint->setWordWrap(true);
        hint->setStyleSheet(QStringLiteral("color: #969aa0;"));
        leftLay->addWidget(hint);

        auto* convertBtn = new QPushButton(QStringLiteral("Convert"));
        convertBtn->setMinimumHeight(32);
        convertBtn_ = convertBtn;
        leftLay->addWidget(convertBtn);
        connect(convertBtn, &QPushButton::clicked, this, &TxConverterWindow::onConvert);
        connect(sourceEdit_, &QLineEdit::editingFinished, this, [this] {
            syncFormatUi();
            syncPipeline();
        });
        connect(outputEdit_, &QLineEdit::editingFinished, this, [this] { syncPipeline(); });

        progressBar_ = new QProgressBar();
        progressBar_->setRange(0, 100);
        progressBar_->setValue(0);
        progressBar_->setTextVisible(true);
        progressBar_->setFormat(QStringLiteral("%p%"));
        progressBar_->setVisible(false);
        leftLay->addWidget(progressBar_);

        progressLabel_ = new QLabel();
        progressLabel_->setWordWrap(true);
        progressLabel_->setStyleSheet(QStringLiteral("color: #a8adb4;"));
        progressLabel_->setVisible(false);
        leftLay->addWidget(progressLabel_);

        leftLay->addStretch(1);

        ocioStatus_ = new QLabel(QStringLiteral("OCIO: checking…"));
        ocioStatus_->setWordWrap(true);
        ocioStatus_->setAlignment(Qt::AlignLeft | Qt::AlignBottom);
        ocioStatus_->setStyleSheet(QStringLiteral("color: #969aa0; padding: 4px 0 0 0;"));
        leftLay->addWidget(ocioStatus_);

        splitter->addWidget(leftPanel);

        auto* viewBox = new QGroupBox(QStringLiteral("Texture Viewer"));
        auto* viewLay = new QVBoxLayout(viewBox);
        viewLay->setContentsMargins(6, 8, 6, 6);
        viewLay->setSpacing(4);
        viewer_ = new sol::TextureViewerWidget(viewBox);
        viewLay->addWidget(viewer_);
        splitter->addWidget(viewBox);

        splitter->setStretchFactor(0, 0);
        splitter->setStretchFactor(1, 1);
        splitter->setSizes({380, 900});

        root->addWidget(splitter, 1);

        status_ = new QLabel(QStringLiteral("Ready."));
        status_->setWordWrap(true);
        root->addWidget(status_);

        connect(viewer_, &sol::TextureViewerWidget::statusMessage, this, [this](const QString& msg) {
            status_->setText(msg);
        });
        syncViewerOcio();
        refreshOcioStatus();
        syncPipeline();
    }

private:
    sol::TxOutputFormat selectedFormat() const {
        return sol::TxOutputFormat(formatCombo_->currentData().toInt());
    }

    sol::TxOutputFormat effectiveFormat() const {
        return sol::txResolveFormat(selectedFormat(), sourceEdit_->text().trimmed().toStdString());
    }

    static bool configLooksLikeAces(const std::string& configPath) {
        if (configPath.empty()) return false;
        const auto spaces = sol::txColorSpacesFromConfig(configPath);
        for (const std::string& s : spaces) {
            const QString lower = QString::fromStdString(s).toLower();
            if (lower.contains(QLatin1String("acescg")) || lower.contains(QLatin1String("aces")))
                return true;
        }
        // Filename heuristic for ACES configs (e.g. aces_1.3/config.ocio).
        const QString pathLower = QString::fromStdString(configPath).toLower();
        return pathLower.contains(QLatin1String("aces"));
    }

    void refreshOcioStatus() {
        if (!ocioStatus_) return;
        const bool useEnv = useEnvCheck_ && useEnvCheck_->isChecked();
        const std::string settings =
            ocioEdit_ ? ocioEdit_->text().trimmed().toStdString() : std::string();
        const sol::OcioStatus st = sol::ocioEnsureConfig(useEnv, settings);
        const bool hasAces = st.configLoaded && configLooksLikeAces(st.configPath);

        if (st.configLoaded && hasAces) {
            const QString where = st.fromEnvironment ? QStringLiteral("OCIO env")
                                                     : QStringLiteral("config path");
            ocioStatus_->setText(
                QStringLiteral("OCIO + ACES: loaded — all good\n%1 [%2]")
                    .arg(QString::fromStdString(st.configPath), where));
            ocioStatus_->setStyleSheet(QStringLiteral("color: #7cbc7c; padding: 4px 0 0 0;"));
            ocioStatus_->setToolTip(QString::fromStdString(st.message));
        } else if (st.configLoaded) {
            ocioStatus_->setText(
                QStringLiteral("OCIO: loaded, but ACES spaces not detected\n%1")
                    .arg(QString::fromStdString(st.configPath)));
            ocioStatus_->setStyleSheet(QStringLiteral("color: #c9a86c; padding: 4px 0 0 0;"));
            ocioStatus_->setToolTip(QString::fromStdString(st.message));
        } else {
            ocioStatus_->setText(QString::fromStdString(
                st.message.empty() ? "OCIO: not found" : st.message));
            ocioStatus_->setStyleSheet(QStringLiteral("color: #c97a7a; padding: 4px 0 0 0;"));
            ocioStatus_->setToolTip(ocioStatus_->text());
        }
    }

    void setConvertProgress(int completed, int total, const QString& currentFile) {
        if (!progressBar_ || !progressLabel_) return;
        const int tot = std::max(0, total);
        const int done = std::clamp(completed, 0, tot);
        progressBar_->setVisible(true);
        progressLabel_->setVisible(true);
        progressBar_->setRange(0, tot > 0 ? tot : 1);
        progressBar_->setValue(done);
        progressBar_->setFormat(QStringLiteral("%v / %m"));
        const int remainingPct =
            tot > 0 ? int(std::lround(100.0 * double(tot - done) / double(tot))) : 0;
        if (tot <= 0) {
            progressLabel_->setText(QStringLiteral("Preparing…"));
        } else if (done >= tot) {
            progressLabel_->setText(QStringLiteral("Finishing… %1/%2").arg(done).arg(tot));
        } else {
            const QString name = currentFile.isEmpty() ? QStringLiteral("…")
                                                       : QFileInfo(currentFile).fileName();
            progressLabel_->setText(
                QStringLiteral("%1% remaining · %2/%3 · %4")
                    .arg(remainingPct)
                    .arg(done)
                    .arg(tot)
                    .arg(name));
        }
        QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    }

    void hideConvertProgress() {
        if (progressBar_) progressBar_->setVisible(false);
        if (progressLabel_) progressLabel_->setVisible(false);
    }

    void syncFormatUi() {
        const auto eff = effectiveFormat();
        const bool isTx = eff == sol::TxOutputFormat::Tx;
        const bool isJpg = eff == sol::TxOutputFormat::Jpg;
        const bool isExr = eff == sol::TxOutputFormat::Exr;
        const bool isTiff = eff == sol::TxOutputFormat::Tiff;
        const bool isPng = eff == sol::TxOutputFormat::Png;

        if (colorSpaceCombo_) colorSpaceCombo_->setVisible(isTx);
        if (colorSpaceLabel_) colorSpaceLabel_->setVisible(isTx);
        if (advancedCheck_) advancedCheck_->setVisible(isTx);
        if (useEnvCheck_) useEnvCheck_->setVisible(isTx);
        if (ocioEdit_) {
            if (QWidget* w = ocioEdit_->parentWidget()) w->setVisible(isTx);
        }
        if (ocioLabel_) ocioLabel_->setVisible(isTx);

        // Bit depth
        const int prevBit = bitDepthCombo_->currentData().isValid() ? bitDepthCombo_->currentData().toInt()
                                                                    : 0;
        bitDepthCombo_->blockSignals(true);
        bitDepthCombo_->clear();
        if (isJpg) {
            bitDepthCombo_->addItem(QStringLiteral("8"), 8);
            if (QWidget* field = form_->labelForField(bitDepthCombo_)) field->setVisible(false);
            bitDepthCombo_->setVisible(false);
        } else {
            if (QWidget* field = form_->labelForField(bitDepthCombo_)) field->setVisible(true);
            bitDepthCombo_->setVisible(true);
            if (isExr || isTiff) {
                bitDepthCombo_->addItem(QStringLiteral("16"), 16);
                bitDepthCombo_->addItem(QStringLiteral("32"), 32);
            } else if (isPng) {
                bitDepthCombo_->addItem(QStringLiteral("8"), 8);
                bitDepthCombo_->addItem(QStringLiteral("16"), 16);
            } else {  // TX
                bitDepthCombo_->addItem(QStringLiteral("8"), 8);
                bitDepthCombo_->addItem(QStringLiteral("16"), 16);
                bitDepthCombo_->addItem(QStringLiteral("32"), 32);
            }
            int prefer = isExr || isTiff || isTx ? 16 : 8;
            if (bitDepthCombo_->findData(prevBit) >= 0) prefer = prevBit;
            const int idx = bitDepthCombo_->findData(prefer);
            bitDepthCombo_->setCurrentIndex(idx >= 0 ? idx : 0);
        }
        bitDepthCombo_->blockSignals(false);

        // Channels — JPG has no alpha.
        const int prevCh = channelsCombo_->currentData().isValid() ? channelsCombo_->currentData().toInt()
                                                                   : int(sol::TxChannelMode::RGBA);
        channelsCombo_->blockSignals(true);
        channelsCombo_->clear();
        if (isJpg) {
            channelsCombo_->addItem(QStringLiteral("RGB"), int(sol::TxChannelMode::RGB));
            channelsCombo_->addItem(QStringLiteral("R"), int(sol::TxChannelMode::R));
            channelsCombo_->addItem(QStringLiteral("G"), int(sol::TxChannelMode::G));
            channelsCombo_->addItem(QStringLiteral("B"), int(sol::TxChannelMode::B));
        } else {
            channelsCombo_->addItem(QStringLiteral("RGBA"), int(sol::TxChannelMode::RGBA));
            channelsCombo_->addItem(QStringLiteral("RGB"), int(sol::TxChannelMode::RGB));
            channelsCombo_->addItem(QStringLiteral("R"), int(sol::TxChannelMode::R));
            channelsCombo_->addItem(QStringLiteral("G"), int(sol::TxChannelMode::G));
            channelsCombo_->addItem(QStringLiteral("B"), int(sol::TxChannelMode::B));
            channelsCombo_->addItem(QStringLiteral("A"), int(sol::TxChannelMode::A));
        }
        int chIdx = channelsCombo_->findData(prevCh);
        if (chIdx < 0) chIdx = 0;
        channelsCombo_->setCurrentIndex(chIdx);
        channelsCombo_->blockSignals(false);

        if (viewer_) {
            const std::string src = sourceEdit_->text().trimmed().toStdString();
            viewer_->setOutputExtension(QString::fromStdString(
                sol::txOutputExtension(selectedFormat(), src)));
        }
    }

    void refreshColorSpaces(bool advanced) {
        const QString current = colorSpaceCombo_->currentText();
        colorSpaceCombo_->clear();
        std::vector<std::string> spaces;
        if (advanced) {
            const std::string cfg =
                sol::txResolveOcioConfig(useEnvCheck_ && useEnvCheck_->isChecked(),
                                         ocioEdit_ ? ocioEdit_->text().toStdString() : std::string());
            spaces = sol::txColorSpacesFromConfig(cfg);
        } else {
            spaces = sol::txCuratedColorSpaces();
        }
        for (const std::string& s : spaces) colorSpaceCombo_->addItem(QString::fromStdString(s));
        const int idx = colorSpaceCombo_->findText(current);
        if (idx >= 0) colorSpaceCombo_->setCurrentIndex(idx);
        else if (!current.isEmpty()) colorSpaceCombo_->setEditText(current);
        else colorSpaceCombo_->setCurrentIndex(0);
    }

    void syncViewerOcio() {
        if (!viewer_) return;
        viewer_->setOcioConfig(useEnvCheck_->isChecked(), ocioEdit_->text().trimmed());
    }

    void syncPipeline() {
        if (!viewer_) return;
        syncViewerOcio();
        const std::string src = sourceEdit_->text().trimmed().toStdString();
        viewer_->setOutputExtension(
            QString::fromStdString(sol::txOutputExtension(selectedFormat(), src)));
        viewer_->setPipelinePaths(sourceEdit_->text().trimmed(), outputEdit_->text().trimmed());
    }

    void onConvert() {
        const QString src = sourceEdit_->text().trimmed();
        const QString outDir = outputEdit_->text().trimmed();
        if (src.isEmpty()) {
            QMessageBox::warning(this, QStringLiteral("TX Converter"),
                                 QStringLiteral("Choose a source texture."));
            return;
        }
        if (outDir.isEmpty()) {
            QMessageBox::warning(this, QStringLiteral("TX Converter"),
                                 QStringLiteral("Choose an output folder before converting."));
            return;
        }

        const auto selected = selectedFormat();
        const auto effective = sol::txResolveFormat(selected, src.toStdString());

        sol::TxConvertOptions opt;
        opt.format = selected;  // Original kept; resolved inside convert
        opt.bitDepth = bitDepthCombo_->isVisible() ? bitDepthCombo_->currentData().toInt() : 8;
        opt.longSide = resolutionCombo_->currentData().toInt();
        opt.channels = sol::TxChannelMode(channelsCombo_->currentData().toInt());
        opt.frameStart = viewer_ ? viewer_->rangeStart() : 1;
        opt.frameEnd = viewer_ ? viewer_->rangeEnd() : 1;
        opt.updateOnly = true;
        opt.memoryBudgetBytes =
            viewer_ ? viewer_->memoryBudgetBytes() : (32LL * 1024 * 1024 * 1024);
        if (effective == sol::TxOutputFormat::Tx) {
            opt.inputColorSpace = colorSpaceCombo_->currentText().toStdString();
            opt.ocioConfigPath = sol::txResolveOcioConfig(useEnvCheck_->isChecked(),
                                                          ocioEdit_->text().trimmed().toStdString());
        }

        auto results = std::make_shared<std::vector<sol::TxConvertResult>>();
        auto error = std::make_shared<std::string>();
        auto progressState = std::make_shared<sol::TxConvertProgressState>();
        auto okFlag = std::make_shared<std::atomic<bool>>(false);

        status_->setText(QStringLiteral("Converting…"));
        if (convertBtn_) convertBtn_->setEnabled(false);
        setConvertProgress(0, 1, {});
        QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);

        auto future = std::async(std::launch::async, [=]() {
            sol::TxConvertProgressFn progress = [progressState](int completed, int total,
                                                                const std::string& /*path*/) {
                progressState->total.store(total);
                progressState->completed.store(completed);
            };
            const bool ok =
                sol::txConvertPattern(src.toStdString(), outDir.toStdString(), opt, *results,
                                      *error, progress);
            okFlag->store(ok);
            return ok;
        });

        while (future.wait_for(std::chrono::milliseconds(33)) != std::future_status::ready) {
            const int tot = progressState->total.load();
            const int done = progressState->completed.load();
            setConvertProgress(done, std::max(1, tot), {});
        }
        future.get();
        const bool ok = okFlag->load();

        hideConvertProgress();
        if (convertBtn_) convertBtn_->setEnabled(true);
        refreshOcioStatus();

        int success = 0;
        for (const auto& r : *results)
            if (r.ok) ++success;
        if (ok) {
            status_->setText(QStringLiteral("Done — %1 file(s) → %2").arg(success).arg(outDir));
            if (viewer_) {
                viewer_->reloadConvertedBuffer();
                viewer_->setContentKind(sol::TextureViewerWidget::ViewerContentKind::ConvertedTx);
            }
        } else {
            status_->setText(QStringLiteral("Finished with errors (%1 ok): %2")
                                 .arg(success)
                                 .arg(QString::fromStdString(*error)));
            QMessageBox::warning(this, QStringLiteral("TX Converter"),
                                 QString::fromStdString(error->empty() ? "conversion failed" : *error));
        }
    }

    QFormLayout* form_ = nullptr;
    QLineEdit* sourceEdit_ = nullptr;
    QLineEdit* outputEdit_ = nullptr;
    QLineEdit* ocioEdit_ = nullptr;
    QComboBox* formatCombo_ = nullptr;
    QComboBox* bitDepthCombo_ = nullptr;
    QComboBox* resolutionCombo_ = nullptr;
    QComboBox* channelsCombo_ = nullptr;
    QComboBox* colorSpaceCombo_ = nullptr;
    QCheckBox* advancedCheck_ = nullptr;
    QCheckBox* useEnvCheck_ = nullptr;
    QWidget* colorSpaceLabel_ = nullptr;
    QWidget* ocioLabel_ = nullptr;
    QPushButton* convertBtn_ = nullptr;
    QProgressBar* progressBar_ = nullptr;
    QLabel* progressLabel_ = nullptr;
    QLabel* ocioStatus_ = nullptr;
    QLabel* status_ = nullptr;
    sol::TextureViewerWidget* viewer_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QImageReader::setAllocationLimit(0);
    sol::applyDarkTheme(app);
    TxConverterWindow window;
    window.show();
    return app.exec();
}
