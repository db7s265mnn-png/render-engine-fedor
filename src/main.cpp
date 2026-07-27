// Bob_Render - node based path tracer with Embree and OptiX backends.
#include <QApplication>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QGuiApplication>
#include <QStringList>

#include "app/headless.h"
#include "core/log.h"
#include "nodes/node_registry.h"
#include "solstice_config.h"
#include "ui/main_window.h"
#include "ui/theme.h"

#ifdef _WIN32
#  include <windows.h>
#  include <cstdio>
#endif

namespace {

// The application is linked as a GUI subsystem binary so that launching it
// never flashes a console window. In headless mode we borrow the console of
// the shell that started us, otherwise the progress output would go nowhere.
void attachParentConsole() {
#ifdef _WIN32
    if (!AttachConsole(ATTACH_PARENT_PROCESS)) return;
    FILE* stream = nullptr;
    freopen_s(&stream, "CONOUT$", "w", stdout);
    freopen_s(&stream, "CONOUT$", "w", stderr);
    freopen_s(&stream, "CONIN$", "r", stdin);
#endif
}

bool wantsHeadless(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const QString argument = QString::fromLocal8Bit(argv[i]);
        if (argument == "--headless" || argument == "-r" || argument == "--render") return true;
    }
    return false;
}

void configureParser(QCommandLineParser& parser) {
    parser.setApplicationDescription(
        SOLSTICE_APP_NAME " - a node based path tracer.\n"
        "Loads Alembic and USD geometry, lights it with area and HDRI dome lights and renders\n"
        "with the Embree CPU backend or the OptiX GPU backend.");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addPositionalArgument("scene", "Scene file to open (.solstice)", "[scene]");
    parser.addOptions({
        {{"r", "headless", "render"}, "Render without opening the user interface."},
        {{"o", "output"}, "Output image (.png, .exr or .hdr).", "path", "render.png"},
        {{"s", "samples"}, "Samples per pixel; overrides the render settings node.", "count"},
        {"width", "Image width override.", "pixels"},
        {"height", "Image height override.", "pixels"},
        {{"b", "backend"}, "Render backend: cpu or gpu.", "name"},
        {{"j", "threads"}, "CPU thread count (0 = all cores).", "count"},
        {{"a", "abc"}, "Alembic file to import when no scene file is given.", "path"},
        {{"e", "hdri"}, "HDRI used by the generated dome light.", "path"},
        {"save-scene", "Write the resulting node network to a .solstice file.", "path"},
        {"no-render", "Skip rendering, useful together with --save-scene."},
        {"verbose", "Verbose logging."},
    });
}

}  // namespace

int main(int argc, char** argv) {
    sol::registerBuiltinNodes();

    if (wantsHeadless(argc, argv)) {
        attachParentConsole();
        QCoreApplication application(argc, argv);
        QCoreApplication::setApplicationName(SOLSTICE_APP_NAME);
        QCoreApplication::setApplicationVersion(SOLSTICE_VERSION);

        QCommandLineParser parser;
        configureParser(parser);
        parser.process(application);

        sol::HeadlessOptions options;
        const QStringList positional = parser.positionalArguments();
        if (!positional.isEmpty()) options.scenePath = positional.first();
        options.alembicPath = parser.value("abc");
        options.hdriPath = parser.value("hdri");
        options.outputPath = parser.value("output");
        if (parser.isSet("samples")) options.samples = parser.value("samples").toInt();
        if (parser.isSet("width")) options.width = parser.value("width").toInt();
        if (parser.isSet("height")) options.height = parser.value("height").toInt();
        if (parser.isSet("threads")) options.threads = parser.value("threads").toInt();
        if (parser.isSet("backend")) {
            const QString backend = parser.value("backend").toLower();
            options.backend = (backend == "gpu" || backend == "optix") ? 1 : 0;
        }
        options.saveScenePath = parser.value("save-scene");
        options.renderImage = !parser.isSet("no-render");
        options.verbose = parser.isSet("verbose");
        return sol::runHeadless(options);
    }

    QApplication application(argc, argv);
    QCoreApplication::setApplicationName(SOLSTICE_APP_NAME);
    QCoreApplication::setOrganizationName("Bob");
    QCoreApplication::setApplicationVersion(SOLSTICE_VERSION);
    QGuiApplication::setApplicationDisplayName(QString::fromUtf8(SOLSTICE_APP_NAME));

    QCommandLineParser parser;
    configureParser(parser);
    parser.process(application);

    sol::applyDarkTheme(application);

    sol::MainWindow window;
    const QStringList positional = parser.positionalArguments();
    if (!positional.isEmpty()) {
        window.openScene(positional.first());
    } else if (parser.isSet("abc")) {
        window.newSceneFromAlembic(parser.value("abc"), parser.value("hdri"));
    } else {
        window.newScene();
    }
    window.show();
    return application.exec();
}
