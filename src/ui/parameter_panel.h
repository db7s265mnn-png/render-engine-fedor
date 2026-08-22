// Builds editing widgets for the parameters of the selected node — either a
// LOP/network node or a MaterialX node inside a material container.
#pragma once

#include <QHash>
#include <QWidget>

#include "nodes/node.h"
#include "ui/material_network_view.h"

class QVBoxLayout;
class QLineEdit;
class QPushButton;

namespace sol {

class ParameterPanel : public QWidget {
    Q_OBJECT

public:
    explicit ParameterPanel(QWidget* parent = nullptr);

    void setNode(Node* node);
    void setMaterialXSelection(const MaterialXSelection& selection);
    void clearSelection();
    Node* node() const { return node_; }
    bool showingMaterialX() const { return materialXMode_; }
    void refresh();
    void setFocusPickActive(bool active);

signals:
    void parameterEdited(sol::Node* node, const QString& parameterName);
    // Button parameters (e.g. Render Settings → Render) — no value change.
    void parameterAction(sol::Node* node, const QString& parameterName);
    void nodeRenamed(sol::Node* node);
    void materialXRenamed(sol::Node* hostMaterial, const QString& oldName, const QString& newName);
    void materialXInputEdited(sol::Node* hostMaterial, const QString& nodeName, const QString& inputName,
                              const QString& value);
    void materialXTypeEdited(sol::Node* hostMaterial, const QString& nodeName, const QString& type);
    // Camera Focus Pick: toggle viewport pick for DOF focus distance.
    void focusPickToggled(bool active);

private:
    void rebuild();
    void rebuildLop();
    void rebuildMaterialX();
    QWidget* createEditor(Parameter& parameter);

    Node* node_ = nullptr;
    bool materialXMode_ = false;
    MaterialXSelection materialX_;
    QWidget* content_ = nullptr;
    QVBoxLayout* contentLayout_ = nullptr;
    QLineEdit* nameEdit_ = nullptr;
    bool updating_ = false;
    bool focusPickActive_ = false;
    QPushButton* focusPickButton_ = nullptr;
    // Last open parameter folder per node type (Houdini-style tabs survive rebuild).
    QHash<QString, QString> lastFolderByType_;
};

}  // namespace sol
