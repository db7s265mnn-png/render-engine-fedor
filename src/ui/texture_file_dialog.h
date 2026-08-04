// Houdini-like texture / file browser with optional UDIM / $F sequence grouping.
#pragma once

#include <QDialog>
#include <QString>
#include <QStringList>

class QCheckBox;
class QComboBox;
class QFileSystemModel;
class QLabel;
class QLineEdit;
class QListView;
class QTreeView;
class QStandardItemModel;

namespace sol {

enum class SequenceTokenKind {
    None = 0,
    Udim,   // <UDIM>
    F,      // $F
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
};

struct TextureFileDialogResult {
    QString path;                 // concrete file or pattern with <UDIM> / $F#
    bool sequence = false;
    SequenceTokenKind token = SequenceTokenKind::None;
};

class TextureFileDialog : public QDialog {
    Q_OBJECT
public:
    explicit TextureFileDialog(QWidget* parent = nullptr);

    void setStartPath(const QString& path);
    void setNameFilters(const QStringList& filters);  // e.g. {"*.png", "*.exr", ...}
    void setWindowTitleText(const QString& title);

    TextureFileDialogResult resultData() const { return result_; }

    // Static helper used by call sites.
    static TextureFileDialogResult getOpenTexture(QWidget* parent, const QString& title,
                                                  const QString& startPath,
                                                  const QString& filter);

private slots:
    void onDirectorySelected(const QModelIndex& index);
    void onFileActivated(const QModelIndex& index);
    void onSequenceToggled(bool on);
    void onTokenChanged(int);
    void onPathEdited();
    void acceptSelection();
    void refreshFileList();

private:
    struct SeqGroup {
        QString displayName;   // foo.<UDIM>.png or foo.$F4.exr
        QString patternPath;   // full path with token
        QStringList members;   // concrete files
        SequenceTokenKind kind = SequenceTokenKind::None;
        int padWidth = 0;
    };

    void populateFiles();
    QVector<SeqGroup> detectSequences(const QStringList& files) const;
    SequenceTokenKind currentTokenKind() const;
    QString tokenString(SequenceTokenKind kind, int padWidth) const;

    QFileSystemModel* dirModel_ = nullptr;
    QTreeView* dirView_ = nullptr;
    QListView* fileView_ = nullptr;
    QStandardItemModel* fileModel_ = nullptr;
    QLineEdit* pathEdit_ = nullptr;
    QCheckBox* sequenceCheck_ = nullptr;
    QComboBox* tokenCombo_ = nullptr;
    QLabel* hintLabel_ = nullptr;
    QStringList nameFilters_;
    TextureFileDialogResult result_;
    QVector<SeqGroup> groups_;
};

}  // namespace sol
