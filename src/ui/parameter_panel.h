// Builds editing widgets for the parameters of the selected node.
#pragma once

#include <QWidget>

#include "nodes/node.h"

class QVBoxLayout;
class QLineEdit;

namespace sol {

class ParameterPanel : public QWidget {
    Q_OBJECT

public:
    explicit ParameterPanel(QWidget* parent = nullptr);

    void setNode(Node* node);
    Node* node() const { return node_; }
    void refresh();

signals:
    void parameterEdited(sol::Node* node, const QString& parameterName);
    void nodeRenamed(sol::Node* node);

private:
    void rebuild();
    QWidget* createEditor(Parameter& parameter);

    Node* node_ = nullptr;
    QWidget* content_ = nullptr;
    QVBoxLayout* contentLayout_ = nullptr;
    QLineEdit* nameEdit_ = nullptr;
    bool updating_ = false;
};

}  // namespace sol
