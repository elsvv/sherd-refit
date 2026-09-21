# Сборка sherd-refit в Blender.
#
# Файл написан приложением — «Открыть в Blender». Он ничего не удаляет и ничего
# не трогает из того, что уже есть в файле: всё складывается в новую коллекцию
# с именем сборки, по коллекции на группу, по объекту на фрагмент.
#
# Позы — те же матрицы, что показывает «Сборка». Масштаб не меняется: единицы
# остаются единицами скана.
#
#     blender --python open_in_blender.py
#
# «karas — сборка»

import os

import bpy
from mathutils import Matrix, Vector

# Операторов `wm.ply_import` и `wm.stl_import` в Blender 3.x нет вовсе, поэтому
# версия проверяется до первого импорта (A §9.2).
MIN_BLENDER = (4, 0, 0)

# Имя верхней коллекции: всё, что делает скрипт, лежит внутри неё.
TITLE = "karas — сборка"

# Импортёр glTF переводит Y-up в Z-up прямо в вершинах: (x, y, z) -> (x, -z, y).
# Наши GLB хранят координаты скана как есть, поэтому поза домножается справа на
# обратное преобразование — иначе группа ляжет набок.
GLTF_TO_SCAN = Matrix((
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
))

# Зазор между группами в ряду — доля ширины предыдущей группы.
GAP = 0.2

GROUPS = [
    {
        "label": "Группа 0",
        "fragments": [
            {
                "name": "FY234008",
                "path": "/Volumes/Архив/сканы керамика 2024/FY234008.ply",
                "importer": "ply",
                "matrix": (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            },
            {
                "name": "FY234012",
                "path": "/Users/museum/Мои \"рабочие\" файлы/karas/fragments/FY234012.glb",
                "importer": "gltf",
                "matrix": (
                    (0.8660254037844386, -0.5, 0.0, 12.5),
                    (0.5, 0.8660254037844386, 0.0, -3.25),
                    (0.0, 0.0, 1.0, 0.125),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            },
        ],
    },
    {
        "label": "Группа 2",
        "fragments": [
            {
                "name": "KP-7",
                "path": "C:\\Сканы\\керамика\\KP-7.STL",
                "importer": "stl",
                "matrix": (
                    (-1.0, 0.0, 0.0, 1e-13),
                    (0.0, 1.0, 0.0, -0.0),
                    (0.0, 0.0, -1.0, 250.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            },
            {
                "name": "KP-8",
                "path": "/Volumes/Архив/сканы керамика 2024/KP-8.obj",
                "importer": "obj",
                "matrix": (
                    (1.0, 0.0, 0.0, -101.0625),
                    (0.0, 0.3333333333333333, -0.6666666666666666, 0.0),
                    (0.0, 0.6666666666666666, 0.3333333333333333, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            },
        ],
    },
]


def _say(message):
    """Консоль Blender — единственное место, где скрипт может говорить подробно."""
    print("[sherd-refit] " + message)


def _popup(lines):
    """То же самое окном, когда окно есть: фоновый Blender обходится консолью."""
    manager = getattr(bpy.context, "window_manager", None)
    if manager is None:
        return

    def draw(self, _context):
        for line in lines:
            self.layout.label(text=line)

    try:
        manager.popup_menu(draw, title="sherd-refit", icon="INFO")
    except Exception as error:
        _say(str(error))


def _import(fragment):
    """Читает один файл и возвращает объекты, которых до него не было."""
    path = fragment["path"]
    if not os.path.isfile(path):
        raise IOError("файла нет: " + path)
    before = set(bpy.data.objects)
    kind = fragment["importer"]
    if kind == "ply":
        bpy.ops.wm.ply_import(filepath=path, forward_axis="Y", up_axis="Z")
    elif kind == "obj":
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    elif kind == "stl":
        bpy.ops.wm.stl_import(filepath=path, forward_axis="Y", up_axis="Z")
    else:
        bpy.ops.import_scene.gltf(filepath=path)
    return [obj for obj in bpy.data.objects if obj not in before]


def _adopt(objects, collection):
    """Переносит импортированное в коллекцию группы, куда бы импортёр его ни положил."""
    for obj in objects:
        for other in list(obj.users_collection):
            if other is not collection:
                other.objects.unlink(obj)
        if collection not in obj.users_collection:
            collection.objects.link(obj)


def _place(fragment, objects):
    """Имя фрагмента и его поза. Масштаб не трогаем: единицы остаются сканными."""
    pose = Matrix(fragment["matrix"])
    if fragment["importer"] == "gltf":
        pose = pose @ GLTF_TO_SCAN
    for obj in objects:
        obj.name = fragment["name"]
        # Вложенным объектам поза не нужна: они едут за своим корнем.
        if obj.parent not in objects:
            obj.matrix_world = pose


def _width(objects):
    """Ширина группы по X в мировых координатах: по ней считается сдвиг следующей."""
    edges = []
    for obj in objects:
        for corner in getattr(obj, "bound_box", ()):
            edges.append((obj.matrix_world @ Vector(corner)).x)
    return max(edges) - min(edges) if edges else 0.0


def build():
    """Собирает сцену и рассказывает, что получилось."""
    if bpy.app.version < MIN_BLENDER:
        message = "Нужен Blender %s или новее, а это %s." % (
            ".".join(str(part) for part in MIN_BLENDER),
            ".".join(str(part) for part in bpy.app.version),
        )
        _say(message)
        _popup([message])
        return

    view_layer = bpy.context.view_layer
    # Импортёры кладут объекты в активную коллекцию. Пусть это будет коллекция
    # сцены, а не чья-то чужая и, может быть, выключенная.
    view_layer.active_layer_collection = view_layer.layer_collection

    top = bpy.data.collections.new(TITLE)
    bpy.context.scene.collection.children.link(top)

    failed = []
    total = 0
    offset = 0.0
    for group in GROUPS:
        collection = bpy.data.collections.new(group["label"])
        top.children.link(collection)
        placed = []
        for fragment in group["fragments"]:
            try:
                objects = _import(fragment)
            except Exception as error:
                failed.append(fragment["name"] + " — " + str(error))
                continue
            if not objects:
                failed.append(fragment["name"] + " — импортёр ничего не добавил")
                continue
            _adopt(objects, collection)
            _place(fragment, objects)
            placed.extend(objects)
        total += len(placed)
        # Ширина считается до переноса: после него мировая матрица объекта
        # зависит от родителя, а тот пересчитывается не сразу.
        width = _width(placed)
        if len(GROUPS) > 1:
            empty = bpy.data.objects.new(group["label"], None)
            empty.empty_display_type = "PLAIN_AXES"
            collection.objects.link(empty)
            for obj in placed:
                if obj.parent is None:
                    obj.parent = empty
            empty.location.x = offset
            offset += width * (1.0 + GAP)

    _say("объектов: %d" % total)
    _say("масштаб не менялся: единицы — те же, что в сканах")
    lines = ["Объектов: %d." % total, "Масштаб не менялся: единицы те же, что в сканах."]
    if failed:
        _say("не удалось прочитать файлов: %d" % len(failed))
        for line in failed:
            _say("  " + line)
        lines.append("Не удалось прочитать: %d (подробности в консоли)." % len(failed))
    _popup(lines)


build()
