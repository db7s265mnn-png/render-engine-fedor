#include "ui/texture_file_dialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileInfo>
#include <QFileSystemModel>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListView>
#include <QPushButton>
#include <QRegularExpression>
#include <QSet>
#include <QSettings>
#include <QSplitter>
#include <QStandardItemModel>
#include <QTreeView>
#include <QVBoxLayout>

#include <algorithm>
#include <map>

namespace sol {
namespace {

constexpr const char* kSettingsOrg = "Bob";
constexpr const char* kSettingsApp = "TextureFileDialog";
constexpr const char* kLastDirKey = "lastDirectory";

QStringList parseFilterGlobs(const QString& filter) {
    QStringList globs;
    static const QRegularExpression re(QStringLiteral(R"(\*\.[A-Za-z0-9]+)"));
    auto it = re.globalMatch(filter);
    while (it.hasNext()) globs << it.next().captured();
    if (globs.isEmpty()) globs << QStringLiteral("*");
    return globs;
}

bool matchGlobs(const QString& fileName, const QStringList& globs) {
    for (const QString& g : globs) {
        if (QDir::match(g, fileName)) return true;
    }
    return false;
}

}  // namespace

TextureFileDialog::TextureFileDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(QStringLiteral("Choose Texture"));
    resize(920, 560);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(8, 8, 8, 8);
    root->setSpacing(6);

    auto* splitter = new QSplitter(Qt::Horizontal, this);

    dirModel_ = new QFileSystemModel(this);
    dirModel_->setFilter(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Drives);
    dirModel_->setRootPath(QStringLiteral(""));
    dirView_ = new QTreeView(splitter);
    dirView_->setModel(dirModel_);
    dirView_->setHeaderHidden(true);
    for (int c = 1; c < dirModel_->columnCount(); ++c) dirView_->hideColumn(c);
    dirView_->setAnimated(false);
    splitter->addWidget(dirView_);

    fileModel_ = new QStandardItemModel(this);
    fileView_ = new QListView(splitter);
    fileView_->setModel(fileModel_);
    fileView_->setSelectionMode(QAbstractItemView::SingleSelection);
    fileView_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    splitter->addWidget(fileView_);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 2);
    root->addWidget(splitter, 1);

    auto* opts = new QHBoxLayout();
    sequenceCheck_ = new QCheckBox(QStringLiteral("Sequence"));
    sequenceCheck_->setToolTip(
        QStringLiteral("Group UDIM / frame sequences into a single entry (min 2 files)."));
    sequenceCheck_->setChecked(true);
    opts->addWidget(sequenceCheck_);
    opts->addWidget(new QLabel(QStringLiteral("Token")));
    tokenCombo_ = new QComboBox();
    tokenCombo_->addItem(QStringLiteral("<UDIM>"), int(SequenceTokenKind::Udim));
    tokenCombo_->addItem(QStringLiteral("$F"), int(SequenceTokenKind::F));
    tokenCombo_->setEnabled(true);
    tokenCombo_->setToolTip(QStringLiteral("Token written into the path when Sequence is on."));
    opts->addWidget(tokenCombo_);
    opts->addStretch(1);
    root->addLayout(opts);

    hintLabel_ = new QLabel(
        QStringLiteral("Sequence on: group tiles/frames and insert <UDIM> or $F. "
                       "Sequence off: load the selected file only."));
    hintLabel_->setWordWrap(true);
    hintLabel_->setStyleSheet(QStringLiteral("color: #9aa0a6;"));
    root->addWidget(hintLabel_);

    pathEdit_ = new QLineEdit();
    root->addWidget(pathEdit_);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);
    root->addWidget(buttons);

    connect(dirView_->selectionModel(), &QItemSelectionModel::currentChanged, this,
            [this](const QModelIndex& cur, const QModelIndex&) { onDirectorySelected(cur); });
    connect(fileView_, &QListView::doubleClicked, this, &TextureFileDialog::onFileActivated);
    connect(fileView_->selectionModel(), &QItemSelectionModel::currentChanged, this,
            [this](const QModelIndex& cur, const QModelIndex&) {
                if (!cur.isValid()) return;
                pathEdit_->setText(cur.data(Qt::UserRole).toString());
            });
    connect(sequenceCheck_, &QCheckBox::toggled, this, &TextureFileDialog::onSequenceToggled);
    connect(tokenCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &TextureFileDialog::onTokenChanged);
    connect(buttons, &QDialogButtonBox::accepted, this, &TextureFileDialog::acceptSelection);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    nameFilters_ = QStringList{QStringLiteral("*")};
}

QString TextureFileDialog::rememberedDirectory() const {
    QSettings settings(QString::fromUtf8(kSettingsOrg), QString::fromUtf8(kSettingsApp));
    const QString dir = settings.value(QString::fromUtf8(kLastDirKey)).toString();
    if (!dir.isEmpty() && QDir(dir).exists()) return dir;
    return {};
}

void TextureFileDialog::rememberDirectory(const QString& dirPath) {
    if (dirPath.isEmpty()) return;
    QFileInfo info(dirPath);
    const QString dir = info.isDir() ? info.absoluteFilePath() : info.absolutePath();
    if (dir.isEmpty() || !QDir(dir).exists()) return;
    QSettings settings(QString::fromUtf8(kSettingsOrg), QString::fromUtf8(kSettingsApp));
    settings.setValue(QString::fromUtf8(kLastDirKey), dir);
}

void TextureFileDialog::setStartPath(const QString& path) {
    QString dir;
    QFileInfo info(path);
    if (info.isFile()) {
        dir = info.absolutePath();
        pathEdit_->setText(info.absoluteFilePath());
    } else if (!path.isEmpty() && QDir(path).exists()) {
        dir = QFileInfo(path).absoluteFilePath();
        pathEdit_->setText(dir);
    } else if (!path.isEmpty()) {
        // May be a pattern with <UDIM> / $F — still use its folder if present.
        dir = info.absolutePath();
        pathEdit_->setText(path);
        if (!QDir(dir).exists()) dir.clear();
    }
    if (dir.isEmpty()) dir = rememberedDirectory();
    if (dir.isEmpty()) dir = QDir::homePath();

    const QModelIndex idx = dirModel_->index(dir);
    if (idx.isValid()) {
        dirView_->setCurrentIndex(idx);
        dirView_->expand(idx);
        dirView_->scrollTo(idx);
        onDirectorySelected(idx);
    }
}

void TextureFileDialog::setNameFilters(const QStringList& filters) {
    nameFilters_ = filters.isEmpty() ? QStringList{QStringLiteral("*")} : filters;
    refreshFileList();
}

void TextureFileDialog::setWindowTitleText(const QString& title) { setWindowTitle(title); }

SequenceTokenKind TextureFileDialog::currentTokenKind() const {
    return SequenceTokenKind(tokenCombo_->currentData().toInt());
}

QString TextureFileDialog::tokenString(SequenceTokenKind kind) {
    switch (kind) {
        case SequenceTokenKind::Udim: return QStringLiteral("<UDIM>");
        case SequenceTokenKind::F: return QStringLiteral("$F");
        default: return QStringLiteral("$F");
    }
}

void TextureFileDialog::onDirectorySelected(const QModelIndex& index) {
    if (!index.isValid()) return;
    rememberDirectory(dirModel_->filePath(index));
    refreshFileList();
}

void TextureFileDialog::onSequenceToggled(bool on) {
    tokenCombo_->setEnabled(on);
    refreshFileList();
}

void TextureFileDialog::onTokenChanged(int) { refreshFileList(); }

void TextureFileDialog::onFileActivated(const QModelIndex& index) {
    if (!index.isValid()) return;
    pathEdit_->setText(index.data(Qt::UserRole).toString());
    acceptSelection();
}

void TextureFileDialog::refreshFileList() { populateFiles(); }

QVector<TextureFileDialog::SeqGroup> TextureFileDialog::detectSequences(
    const QStringList& files) const {
    static const QRegularExpression re(
        QStringLiteral(R"(^(.*?)([._]?)(\d{2,8})(\.[^.]+)$)"));

    struct Key {
        QString prefix;
        QString mid;
        QString suffix;
        int width = 0;
        bool operator<(const Key& o) const {
            if (prefix != o.prefix) return prefix < o.prefix;
            if (mid != o.mid) return mid < o.mid;
            if (suffix != o.suffix) return suffix < o.suffix;
            return width < o.width;
        }
    };

    std::map<Key, QStringList> buckets;
    for (const QString& abs : files) {
        const QFileInfo info(abs);
        const QRegularExpressionMatch m = re.match(info.fileName());
        if (!m.hasMatch()) continue;
        Key key;
        key.prefix = m.captured(1);
        key.mid = m.captured(2);
        key.width = m.captured(3).size();
        key.suffix = m.captured(4);
        buckets[key].push_back(abs);
    }

    const SequenceTokenKind want = currentTokenKind();
    QVector<SeqGroup> out;
    for (auto& [key, members] : buckets) {
        if (members.size() < 2) continue;
        std::sort(members.begin(), members.end());

        bool allUdim = true;
        for (const QString& abs : members) {
            const QRegularExpressionMatch m = re.match(QFileInfo(abs).fileName());
            const int n = m.captured(3).toInt();
            if (n < 1001 || n >= 2000) allUdim = false;
        }

        SeqGroup g;
        g.members = members;
        if (want == SequenceTokenKind::Udim) {
            if (!allUdim) continue;
            g.kind = SequenceTokenKind::Udim;
        } else {
            g.kind = SequenceTokenKind::F;
        }

        const QString tok = tokenString(g.kind);
        const QString displayFile = key.prefix + key.mid + tok + key.suffix;
        g.displayName = displayFile;
        g.patternPath = QDir(QFileInfo(members.first()).absolutePath()).filePath(displayFile);
        out.push_back(g);
    }
    return out;
}

void TextureFileDialog::populateFiles() {
    fileModel_->clear();
    groups_.clear();

    const QModelIndex dirIndex = dirView_->currentIndex();
    if (!dirIndex.isValid()) return;
    const QString dirPath = dirModel_->filePath(dirIndex);
    QDir dir(dirPath);
    const QStringList entries = dir.entryList(QDir::Files | QDir::Readable, QDir::Name);

    QStringList matched;
    matched.reserve(entries.size());
    for (const QString& name : entries) {
        if (!matchGlobs(name, nameFilters_)) continue;
        matched << dir.absoluteFilePath(name);
    }

    if (sequenceCheck_->isChecked()) {
        groups_ = detectSequences(matched);
        QSet<QString> grouped;
        for (const SeqGroup& g : groups_) {
            for (const QString& m : g.members) grouped.insert(m);
            auto* item = new QStandardItem(g.displayName +
                                           QStringLiteral("  (%1)").arg(g.members.size()));
            item->setData(g.patternPath, Qt::UserRole);
            item->setData(1, Qt::UserRole + 1);
            item->setToolTip(g.patternPath);
            fileModel_->appendRow(item);
        }
        for (const QString& abs : matched) {
            if (grouped.contains(abs)) continue;
            auto* item = new QStandardItem(QFileInfo(abs).fileName());
            item->setData(abs, Qt::UserRole);
            item->setData(0, Qt::UserRole + 1);
            fileModel_->appendRow(item);
        }
    } else {
        for (const QString& abs : matched) {
            auto* item = new QStandardItem(QFileInfo(abs).fileName());
            item->setData(abs, Qt::UserRole);
            item->setData(0, Qt::UserRole + 1);
            fileModel_->appendRow(item);
        }
    }
}

void TextureFileDialog::acceptSelection() {
    QString path = pathEdit_->text().trimmed();
    if (path.isEmpty()) {
        const QModelIndex cur = fileView_->currentIndex();
        if (cur.isValid()) path = cur.data(Qt::UserRole).toString();
    }
    if (path.isEmpty()) return;

    result_.path = path;
    result_.sequence = sequenceCheck_->isChecked() &&
                       (path.contains(QLatin1String("<UDIM>")) || path.contains(QLatin1Char('$')));
    result_.token = result_.sequence ? currentTokenKind() : SequenceTokenKind::None;
    rememberDirectory(path);
    accept();
}

TextureFileDialogResult TextureFileDialog::getOpenTexture(QWidget* parent, const QString& title,
                                                          const QString& startPath,
                                                          const QString& filter) {
    TextureFileDialog dialog(parent);
    dialog.setWindowTitleText(title);
    dialog.setNameFilters(parseFilterGlobs(filter));
    // Prefer explicit startPath; otherwise last remembered folder.
    dialog.setStartPath(startPath);
    if (dialog.exec() != QDialog::Accepted) return {};
    return dialog.resultData();
}

}  // namespace sol
