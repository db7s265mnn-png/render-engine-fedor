# Solstice — нодовый path tracer

Самостоятельное настольное приложение на Qt для рендеринга методом трассировки путей.
Загружает Alembic-кэши, расставляет источники света и HDRI-окружение и считает картинку
либо на CPU через Intel Embree, либо на GPU через NVIDIA OptiX. Работа со сценой построена
на нодовой сети в духе Houdini Solaris.

![Интерфейс Solstice](images/ui.png)

## Возможности

* **Импорт Alembic** — полигональные меши и клетки сабдивов из `.abc`, вместе с иерархией
  трансформаций, UV и нормалями, на любой момент времени архива.
* **Свет** — dome (HDRI), distant (солнце), rect, disk и sphere. У всех есть интенсивность,
  экспозиция, форма, нормализация по площади и флаг видимости камерой.
* **HDRI-окружение** — `.hdr`, `.exr` и обычные изображения; карта сэмплируется по важности
  через двумерное CDF, поэтому солнце в небе сходится быстро.
* **Path tracing** — прогрессивный трассировщик путей с NEE, MIS, «принципиальным» BSDF
  (диффуз, GGX-металл/диэлектрик, шероховатое пропускание, эмиссия), русской рулеткой и
  ограничением засветок.
* **Два бэкенда** — Embree 4 (CPU, тайловый пул потоков) и OptiX (GPU, GAS на меш и общий
  IAS). Интегратор, BSDF и сэмплирование света — один и тот же код, компилируемый и для
  CPU, и для CUDA.
* **Нодовая сеть** — каждая нода правит «стейдж», который течёт по сети: геометрия,
  трансформации, назначение материалов по маске примитивов, свет, камера, настройки рендера.
  Есть флаги display и bypass, меню по `Tab` и дерево сцены.
* **Интерактивный рендер** — правки автоматически перекуковывают сеть и перезапускают
  рендер; в вьюпорте навигация в стиле Houdini (Alt + мышь).
* **Пакетный рендер** — тот же исполняемый файл считает кадры из командной строки.

## Сборка

### Linux

```bash
sudo apt install build-essential cmake ninja-build qt6-base-dev libembree-dev \
                 libopenexr-dev libimath-dev

# Alembic обычно не пакетируется, собираем из исходников:
git clone --depth 1 --branch 1.8.6 https://github.com/alembic/alembic.git
cmake -S alembic -B alembic/build -DCMAKE_BUILD_TYPE=Release -DUSE_HDF5=OFF \
      -DALEMBIC_SHARED_LIBS=ON -DUSE_TESTS=OFF
cmake --build alembic/build -j && sudo cmake --install alembic/build

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/Solstice
```

### Windows (готовый `Solstice.exe`)

Нужны Visual Studio 2022, Qt 6 и vcpkg:

```powershell
vcpkg install embree:x64-windows alembic:x64-windows openexr:x64-windows
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
      -DCMAKE_PREFIX_PATH=C:/Qt/6.7.2/msvc2019_64
cmake --build build --config Release
cmake --build build --target deploy
```

После `deploy` папка `build\bin\Release` самодостаточна: её можно заархивировать и
запускать на другой машине без установки. Подробности — в
[building_windows.md](building_windows.md).

### Включить OptiX

```bash
cmake -S . -B build -DSOLSTICE_ENABLE_OPTIX=ON -DOptiX_ROOT=/путь/к/OptiX-SDK-9.0.0
```

Device-программы компилируются в PTX через `nvcc` и зашиваются внутрь исполняемого файла.
Если сборка без OptiX или в системе нет CUDA-устройства, приложение пишет предупреждение и
автоматически переключается на Embree.

## Как пользоваться

| Панель | Назначение |
| --- | --- |
| Network Editor | нодовая сеть; рендерится нода с синим флагом display |
| Render View | прогрессивный результат, Alt+ЛКМ орбита, СКМ панорама, колесо — наезд |
| Parameters | параметры выбранной ноды |
| Scene Graph | примитивы, полученные после кука сети |
| Log | сообщения загрузчиков, кука и рендера |

Горячие клавиши: `Tab` — добавить ноду, `D` — флаг display, `B` — bypass, `F` — вписать сеть,
`Del` — удалить, `F5` — рендер, `Esc` — стоп, `Ctrl+E` — сохранить изображение.

### Типовой сценарий: Alembic + HDRI

1. `Tab` → **Alembic Import**, указать файл `.abc`.
2. `Tab` → **Material**, в поле *Assign To* написать `*` или маску вида `/geo/body*`.
3. `Tab` → **Dome Light**, выбрать `.hdr` или `.exr`.
4. `Tab` → **Distant Light** для солнца или **Rect Light** для студийного ключа.
5. `Tab` → **Camera**, затем *Render → Copy View To Camera Node*, чтобы зафиксировать ракурс.
6. `Tab` → **Render Settings**: разрешение, количество сэмплов, бэкенд; поставить на неё
   флаг display.

Ноды соединяются сверху вниз: каждая берёт стейдж со входа, дополняет его и передаёт дальше —
как цепочка LOP-нод в Solaris.

### Командная строка

```bash
# отрендерить существующую сеть
Solstice --headless scene.solstice -o beauty.exr -s 512

# собрать сеть из Alembic и HDRI и сразу посчитать кадр
Solstice --headless -a cache.abc -e studio.hdr -o render.png -s 256 --width 1920 --height 1080

# только сохранить сеть в файл сцены
Solstice --headless -a cache.abc -e studio.hdr --no-render --save-scene shot.solstice
```

Формат вывода определяется расширением: `.png` тонмапится настройками плёнки, `.exr` и `.hdr`
сохраняются в линейном пространстве.

## Примеры

```bash
./build/bin/sol_make_test_abc examples/kit.abc      # тестовый Alembic-архив
python3 tools/make_test_hdri.py examples/sky.hdr    # синтетическое HDRI-небо
./build/bin/Solstice examples/alembic_hdri.solstice
```

## Тесты

```bash
cmake --build build --target solstice_tests && ./build/bin/solstice_tests
```

Проверяются матрицы и сэмплирование, ориентация и pdf карты окружения, энергия BSDF и
согласованность sample/eval, маски примитивов, кук и сериализация сети, а также два сквозных
рендера.

## Ограничения

* Сабдивы рендерятся как исходная полигональная клетка, без Catmull-Clark.
* Кривые, точки и NuPatch из Alembic пропускаются.
* Текстурных карт в материалах пока нет — цвет задаётся параметрами ноды.
* GPU-бэкенд собирается и проверяется в CI, но для работы нужна видеокарта NVIDIA; без неё
  приложение прозрачно использует Embree.
