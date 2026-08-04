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
#include <QImageReader>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSplitter>
#include <QVBoxLayout>
#include <QWidget>

#include "core/log.h"
#include "core/expr_eval.h"
#include "io/image_io.h"
#include "io/tx_convert.h"
#include "texture_viewer.h"
#include "ui/texture_file_dialog.h"
#include "ui/theme.h"

namespace {

class TxConverterWindow : public QWidget {
public:
    TxConverterWindow() {
        setWindowTitle(QStringLiteral("TX Converter"));
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
                    syncPipeline();
                }
            });
            return row;
        };

        form_->addRow(QStringLiteral("Source"), makeBrowseRow(&sourceEdit_, false));
        sourceEdit_->setPlaceholderText(QStringLiteral("texture.png or tile_<UDIM>.exr / .tx"));
        form_->addRow(QStringLiteral("Output Folder"), makeBrowseRow(&outputEdit_, true));
        outputEdit_->setText(QStringLiteral("tx_cache"));

        formatCombo_ = new QComboBox();
        formatCombo_->addItem(QStringLiteral("TX"), int(sol::TxOutputFormat::Tx));
        formatCombo_->addItem(QStringLiteral("PNG"), int(sol::TxOutputFormat::Png));
        formatCombo_->addItem(QStringLiteral("JPG"), int(sol::TxOutputFormat::Jpg));
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
        channelsCombo_->addItem(QStringLiteral("RGBA"), int(sol::TxChannelMode::RGBA));
        channelsCombo_->addItem(QStringLiteral("RGB"), int(sol::TxChannelMode::RGB));
        channelsCombo_->addItem(QStringLiteral("R"), int(sol::TxChannelMode::R));
        channelsCombo_->addItem(QStringLiteral("G"), int(sol::TxChannelMode::G));
        channelsCombo_->addItem(QStringLiteral("B"), int(sol::TxChannelMode::B));
        channelsCombo_->addItem(QStringLiteral("A"), int(sol::TxChannelMode::A));
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
        });
        connect(ocioEdit_, &QLineEdit::editingFinished, this, [this] { syncViewerOcio(); });
        syncOcioEnabled();

        connect(formatCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                [this](int) {
                    syncFormatUi();
                    syncPipeline();
                });
        syncFormatUi();

        leftLay->addWidget(formBox);

        auto* hint = new QLabel(
            QStringLiteral("TX output is ACEScg. PNG/JPG keep source colour (resize/bit/channels only). "
                           "UDIM/$F timeline uses numbers from filenames. "
                           "Viewer: channels + Classic/OCIO view. Wheel = zoom, drag = pan."));
        hint->setWordWrap(true);
        hint->setStyleSheet(QStringLiteral("color: #969aa0;"));
        leftLay->addWidget(hint);

        auto* convertBtn = new QPushButton(QStringLiteral("Convert"));
        convertBtn->setMinimumHeight(32);
        leftLay->addWidget(convertBtn);
        connect(convertBtn, &QPushButton::clicked, this, &TxConverterWindow::onConvert);
        connect(sourceEdit_, &QLineEdit::editingFinished, this, [this] { syncPipeline(); });
        connect(outputEdit_, &QLineEdit::editingFinished, this, [this] { syncPipeline(); });

        leftLay->addStretch(1);
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
        syncPipeline();
    }

private:
    sol::TxOutputFormat currentFormat() const {
        return sol::TxOutputFormat(formatCombo_->currentData().toInt());
    }

    void syncFormatUi() {
        const auto fmt = currentFormat();
        const bool isTx = fmt == sol::TxOutputFormat::Tx;
        const bool isJpg = fmt == sol::TxOutputFormat::Jpg;

        if (colorSpaceCombo_) colorSpaceCombo_->setVisible(isTx);
        if (colorSpaceLabel_) colorSpaceLabel_->setVisible(isTx);
        if (advancedCheck_) advancedCheck_->setVisible(isTx);
        if (useEnvCheck_) useEnvCheck_->setVisible(isTx);
        if (ocioEdit_) {
            // parent row widget
            if (QWidget* w = ocioEdit_->parentWidget()) w->setVisible(isTx);
        }
        if (ocioLabel_) ocioLabel_->setVisible(isTx);

        bitDepthCombo_->blockSignals(true);
        bitDepthCombo_->clear();
        if (isJpg) {
            bitDepthCombo_->addItem(QStringLiteral("8"), 8);
            if (QWidget* field = form_->labelForField(bitDepthCombo_)) field->setVisible(false);
            bitDepthCombo_->setVisible(false);
        } else {
            if (QWidget* field = form_->labelForField(bitDepthCombo_)) field->setVisible(true);
            bitDepthCombo_->setVisible(true);
            bitDepthCombo_->addItem(QStringLiteral("8"), 8);
            bitDepthCombo_->addItem(QStringLiteral("16"), 16);
            if (isTx) bitDepthCombo_->addItem(QStringLiteral("32"), 32);
            const int prefer = isTx ? 16 : 8;
            const int idx = bitDepthCombo_->findData(prefer);
            bitDepthCombo_->setCurrentIndex(idx >= 0 ? idx : 0);
        }
        bitDepthCombo_->blockSignals(false);

        if (viewer_) {
            viewer_->setOutputExtension(QString::fromStdString(sol::txOutputExtension(fmt)));
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
        viewer_->setOutputExtension(QString::fromStdString(sol::txOutputExtension(currentFormat())));
        viewer_->setPipelinePaths(sourceEdit_->text().trimmed(), outputEdit_->text().trimmed());
    }

    void onConvert() {
        const QString src = sourceEdit_->text().trimmed();
        const QString outDir = outputEdit_->text().trimmed();
        if (src.isEmpty() || outDir.isEmpty()) {
            QMessageBox::warning(this, QStringLiteral("TX Converter"),
                                 QStringLiteral("Choose a source texture and an output folder."));
            return;
        }

        sol::TxConvertOptions opt;
        opt.format = currentFormat();
        opt.bitDepth = bitDepthCombo_->isVisible() ? bitDepthCombo_->currentData().toInt() : 8;
        opt.longSide = resolutionCombo_->currentData().toInt();
        opt.channels = sol::TxChannelMode(channelsCombo_->currentData().toInt());
        opt.frameStart = viewer_ ? viewer_->rangeStart() : 1;
        opt.frameEnd = viewer_ ? viewer_->rangeEnd() : 1;
        opt.updateOnly = true;
        if (opt.format == sol::TxOutputFormat::Tx) {
            opt.inputColorSpace = colorSpaceCombo_->currentText().toStdString();
            opt.ocioConfigPath = sol::txResolveOcioConfig(useEnvCheck_->isChecked(),
                                                          ocioEdit_->text().trimmed().toStdString());
        }

        std::vector<sol::TxConvertResult> results;
        std::string error;
        status_->setText(QStringLiteral("Converting…"));
        QApplication::processEvents();
        const bool ok = sol::txConvertPattern(src.toStdString(), outDir.toStdString(), opt, results,
                                              error);
        int success = 0;
        for (const auto& r : results)
            if (r.ok) ++success;
        if (ok) {
            status_->setText(QStringLiteral("Done — %1 file(s) → %2").arg(success).arg(outDir));
            viewer_->setContentKind(sol::TextureViewerWidget::ViewerContentKind::ConvertedTx);
        } else {
            status_->setText(QStringLiteral("Finished with errors (%1 ok): %2")
                                 .arg(success)
                                 .arg(QString::fromStdString(error)));
            QMessageBox::warning(this, QStringLiteral("TX Converter"),
                                 QString::fromStdString(error.empty() ? "conversion failed" : error));
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
