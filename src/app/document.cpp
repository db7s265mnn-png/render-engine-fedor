#include "app/document.h"

#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonParseError>

#include "nodes/node_registry.h"

namespace sol {

bool saveGraphToFile(const NodeGraph& graph, const QString& path, QString& error) {
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        error = "cannot write " + path + ": " + file.errorString();
        return false;
    }
    const QJsonDocument document(graph.toJson());
    file.write(document.toJson(QJsonDocument::Indented));
    file.close();
    return true;
}

bool loadGraphFromFile(NodeGraph& graph, const QString& path, QString& error) {
    registerBuiltinNodes();
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        error = "cannot read " + path + ": " + file.errorString();
        return false;
    }
    const QByteArray data = file.readAll();
    file.close();

    QJsonParseError parseError{};
    const QJsonDocument document = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        error = "invalid scene file: " + parseError.errorString();
        return false;
    }
    QString loadError;
    if (!graph.fromJson(document.object(), loadError)) {
        error = loadError;
        return false;
    }
    graph.setFilePath(QFileInfo(path).absoluteFilePath());
    return true;
}

}  // namespace sol
