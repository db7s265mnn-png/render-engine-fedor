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
#include "io/image_io.h"
#include "io/tx_convert.h"
#include "texture_viewer.h"
#include "ui/theme.h"

namespace {

class TxConverterWindow : public QWidget {
public:
    TxConverterWindow() {
        setWindowTitle(QStringLiteral("TX Converter"));
        resize(980, 720);

        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(10, 10, 10, 10);
        root->setSpacing(8);

        auto* splitter = new QSplitter(Qt::Vertical, this);
        splitter->setChildrenCollapsible(false);

        // ---- Convert panel ----
        auto* convertPanel = new QWidget();
        auto* convertLay = new QVBoxLayout(convertPanel);
        convertLay->setContentsMargins(0, 0, 0, 0);

        auto* formBox = new QGroupBox(QStringLiteral("Convert"));
        auto* form = new QFormLayout(formBox);

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
                    path = QFileDialog::getOpenFileName(
                        this, QStringLiteral("Choose texture"), edit->text(),
                        QStringLiteral(
                            "Images (*.png *.jpg *.jpeg *.exr *.hdr *.tif *.tiff *.bmp *.tx);;All (*)"));
                }
                if (!path.isEmpty()) {
                    edit->setText(path);
                    if (!directory) previewSource();
                }
            });
            return row;
        };

        form->addRow(QStringLiteral("Source"), makeBrowseRow(&sourceEdit_, false));
        sourceEdit_->setPlaceholderText(QStringLiteral("texture.png or tile_<UDIM>.exr / .tx"));
        form->addRow(QStringLiteral("Output Folder"), makeBrowseRow(&outputEdit_, true));
        outputEdit_->setText(QStringLiteral("tx_cache"));

        colorSpaceCombo_ = new QComboBox();
        colorSpaceCombo_->setEditable(true);
        refreshColorSpaces(false);
        form->addRow(QStringLiteral("Color Space"), colorSpaceCombo_);

        advancedCheck_ = new QCheckBox(QStringLiteral("Advanced (full OCIO list)"));
        form->addRow(QString(), advancedCheck_);
        connect(advancedCheck_, &QCheckBox::toggled, this, [this](bool on) { refreshColorSpaces(on); });

        useEnvCheck_ = new QCheckBox(QStringLiteral("Use OCIO from Environment"));
        useEnvCheck_->setChecked(true);
        useEnvCheck_->setToolTip(QStringLiteral("Read config path from the OCIO environment variable."));
        form->addRow(QString(), useEnvCheck_);

        ocioEdit_ = new QLineEdit();
        auto* ocioRow = new QWidget();
        auto* ocioLay = new QHBoxLayout(ocioRow);
        ocioLay->setContentsMargins(0, 0, 0, 0);
        auto* ocioBrowse = new QPushButton(QStringLiteral("…"));
        ocioBrowse->setFixedWidth(32);
        ocioLay->addWidget(ocioEdit_, 1);
        ocioLay->addWidget(ocioBrowse);
        form->addRow(QStringLiteral("OCIO Config"), ocioRow);
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
        connect(useEnvCheck_, &QCheckBox::toggled, this, [syncOcioEnabled](bool) { syncOcioEnabled(); });
        syncOcioEnabled();

        convertLay->addWidget(formBox);

        auto* hint = new QLabel(
            QStringLiteral("Output is always ACEScg (Arnold-style). "
                           "UDIM: put <UDIM> in the source path to convert the whole sequence. "
                           "Preview timeline length = number of UDIM tiles on disk."));
        hint->setWordWrap(true);
        hint->setStyleSheet(QStringLiteral("color: #969aa0;"));
        convertLay->addWidget(hint);

        auto* btnRow = new QHBoxLayout();
        auto* previewBtn = new QPushButton(QStringLiteral("Preview Source"));
        auto* convertBtn = new QPushButton(QStringLiteral("Convert"));
        convertBtn->setMinimumHeight(32);
        previewBtn->setMinimumHeight(32);
        btnRow->addWidget(previewBtn);
        btnRow->addWidget(convertBtn, 1);
        convertLay->addLayout(btnRow);
        connect(previewBtn, &QPushButton::clicked, this, [this] { previewSource(); });
        connect(convertBtn, &QPushButton::clicked, this, &TxConverterWindow::onConvert);
        connect(sourceEdit_, &QLineEdit::editingFinished, this, [this] {
            if (!sourceEdit_->text().trimmed().isEmpty()) previewSource();
        });

        splitter->addWidget(convertPanel);

        // ---- Viewer panel ----
        auto* viewBox = new QGroupBox(QStringLiteral("Texture Viewer"));
        auto* viewLay = new QVBoxLayout(viewBox);
        viewer_ = new sol::TextureViewerWidget(viewBox);
        viewLay->addWidget(viewer_);

        auto* viewBtnRow = new QHBoxLayout();
        auto* openViewBtn = new QPushButton(QStringLiteral("Open in Viewer…"));
        auto* previewOutBtn = new QPushButton(QStringLiteral("Preview Output Folder .tx"));
        viewBtnRow->addWidget(openViewBtn);
        viewBtnRow->addWidget(previewOutBtn);
        viewBtnRow->addStretch(1);
        viewLay->addLayout(viewBtnRow);
        connect(openViewBtn, &QPushButton::clicked, this, [this] {
            const QString path = QFileDialog::getOpenFileName(
                this, QStringLiteral("Preview texture"), sourceEdit_->text(),
                QStringLiteral(
                    "Images (*.png *.jpg *.jpeg *.exr *.hdr *.tif *.tiff *.bmp *.tx);;All (*)"));
            if (!path.isEmpty()) {
                viewer_->setSourcePath(path);
            }
        });
        connect(previewOutBtn, &QPushButton::clicked, this, [this] { previewOutputTx(); });
        connect(viewer_, &sol::TextureViewerWidget::statusMessage, this, [this](const QString& msg) {
            status_->setText(msg);
        });

        splitter->addWidget(viewBox);
        splitter->setStretchFactor(0, 0);
        splitter->setStretchFactor(1, 1);
        splitter->setSizes({280, 440});

        root->addWidget(splitter, 1);

        status_ = new QLabel(QStringLiteral("Ready."));
        status_->setWordWrap(true);
        root->addWidget(status_);
    }

private:
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

    void previewSource() {
        const QString src = sourceEdit_->text().trimmed();
        if (src.isEmpty()) {
            status_->setText(QStringLiteral("Set a Source path to preview."));
            return;
        }
        viewer_->setSourcePath(src);
    }

    void previewOutputTx() {
        const QString src = sourceEdit_->text().trimmed();
        const QString outDir = outputEdit_->text().trimmed();
        if (src.isEmpty() || outDir.isEmpty()) {
            QMessageBox::information(this, QStringLiteral("TX Converter"),
                                     QStringLiteral("Set Source and Output Folder first."));
            return;
        }
        // Prefer converted .tx named like the source basename (or UDIM pattern → .tx).
        QString pattern = src;
        // If source is an image, guess .tx next to output dir with same stem pattern.
        QFileInfo info(src);
        QString name = info.fileName();
        // Replace extension with .tx for single file; keep <UDIM> for sequences.
        if (sol::pathHasUdimToken(name)) {
            const int dot = name.lastIndexOf(QLatin1Char('.'));
            if (dot > 0) name = name.left(dot) + QStringLiteral(".tx");
            else name += QStringLiteral(".tx");
        } else if (sol::pathHasUdimToken(src)) {
            name = info.fileName();
            const int dot = name.lastIndexOf(QLatin1Char('.'));
            if (dot > 0) name = name.left(dot) + QStringLiteral(".tx");
        } else {
            // Concrete UDIM tile or single file → resolve pattern then .tx
            QString udimPattern;
            std::vector<int> tiles;
            if (sol::resolveUdimPattern(src, QString(), udimPattern, tiles)) {
                QFileInfo pinfo(udimPattern);
                name = pinfo.fileName();
                const int dot = name.lastIndexOf(QLatin1Char('.'));
                if (dot > 0) name = name.left(dot) + QStringLiteral(".tx");
            } else {
                name = info.completeBaseName() + QStringLiteral(".tx");
            }
        }
        const QString txPath = QDir(outDir).filePath(name);
        viewer_->setSourcePath(txPath);
    }

    void onConvert() {
        const QString src = sourceEdit_->text().trimmed();
        const QString outDir = outputEdit_->text().trimmed();
        if (src.isEmpty() || outDir.isEmpty()) {
            QMessageBox::warning(this, QStringLiteral("TX Converter"),
                                 QStringLiteral("Choose a source texture and an output folder."));
            return;
        }
        const std::string ocio = sol::txResolveOcioConfig(useEnvCheck_->isChecked(),
                                                          ocioEdit_->text().trimmed().toStdString());
        std::vector<sol::TxConvertResult> results;
        std::string error;
        status_->setText(QStringLiteral("Converting…"));
        QApplication::processEvents();
        const bool ok = sol::txConvertPattern(src.toStdString(), outDir.toStdString(),
                                              colorSpaceCombo_->currentText().toStdString(), ocio,
                                              results, error);
        int success = 0;
        for (const auto& r : results)
            if (r.ok) ++success;
        if (ok) {
            status_->setText(QStringLiteral("Done — %1 file(s) → %2").arg(success).arg(outDir));
            previewOutputTx();
        } else {
            status_->setText(QStringLiteral("Finished with errors (%1 ok): %2")
                                 .arg(success)
                                 .arg(QString::fromStdString(error)));
            QMessageBox::warning(this, QStringLiteral("TX Converter"),
                                 QString::fromStdString(error.empty() ? "conversion failed" : error));
        }
    }

    QLineEdit* sourceEdit_ = nullptr;
    QLineEdit* outputEdit_ = nullptr;
    QLineEdit* ocioEdit_ = nullptr;
    QComboBox* colorSpaceCombo_ = nullptr;
    QCheckBox* advancedCheck_ = nullptr;
    QCheckBox* useEnvCheck_ = nullptr;
    QLabel* status_ = nullptr;
    sol::TextureViewerWidget* viewer_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    // Qt 6 default image allocation limit is 256 MB; 8K UDIM tiles exceed it.
    QImageReader::setAllocationLimit(0);
    sol::applyDarkTheme(app);
    TxConverterWindow window;
    window.show();
    return app.exec();
}
