"""The `ffgs` command: list, add, download, remove and run models.

A thin layer over the library: `ffgs.store` (the records of models, see there),
`GSPipeline.from_pretrained` and `ffgs.io`.

    ffgs ls [--zoo] [--json]
    ffgs add | pull <source>... [--revision R] [--subfolder S]
    ffgs rm <source>...
    ffgs run <source> <image dir> -o <out dir> [--cameras <camera file>]
             [--num-views N] [--revision R] [--subfolder S] [--device D]

A source is anything `GSPipeline.from_pretrained` takes: a local directory, a zoo
name or a Hub repo id. Exit status: 0 done, 1 failed, 2 bad usage.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from .. import __version__, store, zoo

DEFAULT_NUM_VIEWS = 8
# How far an image's aspect ratio may be from its camera's (a resized copy keeps
# it up to rounding).
ASPECT_TOLERANCE = 0.02


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if getattr(args, "num_views", None) is not None and args.num_views < 1:
        parser.error(f"--num-views must be at least 1, got {args.num_views}")
    try:
        args.command(args)
    except KeyboardInterrupt:
        print("ffgs: interrupted", file=sys.stderr)
        return 130
    except Exception as err:  # one line, unless asked for the traceback
        if args.debug:
            traceback.print_exc()
        else:
            message = str(err).strip().splitlines()[0] if str(err).strip() else ""
            print(f"ffgs: error: {message or type(err).__name__}", file=sys.stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffgs", description="Feed-forward 3D Gaussian Splatting models."
    )
    parser.add_argument("--version", action="version", version=f"ffgs {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--debug", action="store_true", help="print the traceback of an error"
    )
    parser.set_defaults(debug=False)
    commands = parser.add_subparsers(title="commands", metavar="<command>")
    commands.required = True

    ls = commands.add_parser(
        "ls", parents=[common], help="list the recorded models (--zoo: the zoo)"
    )
    ls.add_argument("--zoo", action="store_true", help="list the model zoo instead")
    ls.add_argument("--json", action="store_true", help="print JSON")
    ls.set_defaults(command=_ls)

    for name, command, text in (
        ("add", _add, "record models, without downloading them"),
        ("pull", _pull, "download models (recording them if new)"),
        ("rm", _rm, "remove the records of models (never their files)"),
    ):
        sub = commands.add_parser(name, parents=[common], help=text)
        sub.add_argument("sources", nargs="+", metavar="source")
        if name != "rm":
            _source_options(sub)
        sub.set_defaults(command=command)

    run = commands.add_parser(
        "run", parents=[common], help="reconstruct a folder of images"
    )
    run.add_argument("source")
    run.add_argument("images", type=Path, help="folder of .png / .jpg images")
    run.add_argument("-o", "--out", type=Path, required=True, help="output folder")
    run.add_argument(
        "--cameras",
        type=Path,
        help="their cameras: a transforms.json, a COLMAP model directory or an "
        "ffgs cameras.json",
    )
    run.add_argument(
        "--num-views",
        type=int,
        help=f"input images, evenly spaced (default: {DEFAULT_NUM_VIEWS})",
    )
    _source_options(run)
    run.add_argument("--device", help="default: cuda if available, else cpu")
    run.set_defaults(command=_run)
    return parser


def _source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--revision", help="Hub branch, tag or commit")
    parser.add_argument("--subfolder", help="a variant saved in a subfolder")


# --- commands -----------------------------------------------------------------------


def _ls(args: argparse.Namespace) -> None:
    rows = _zoo_rows() if args.zoo else _record_rows()
    if args.json:
        print(json.dumps(rows, indent=1))
        return
    if args.zoo:
        columns = [
            ("NAME", "name"),
            ("MODEL", "model"),
            ("CAMERAS", "camera_input"),
            ("RECORDED", "recorded"),
            ("STATUS", "status"),
            ("SIZE", "size"),
            ("LICENSE", "license"),
        ]
    else:
        if not rows:
            print(
                "no models recorded (`ffgs add <source>`, or run one); "
                "the zoo: `ffgs ls --zoo`"
            )
            return
        columns = [
            ("SOURCE", "source"),
            ("TYPE", "type"),
            ("REVISION", "revision"),
            ("STATUS", "status"),
            ("SIZE", "size"),
            ("LAST USED", "last_used"),
        ]
    _table(rows, columns)


def _add(args: argparse.Namespace) -> None:
    for source in args.sources:
        record = store.add(source, **_options(args))
        print(f"added {_label(record)} ({store.status(record).state})")


def _pull(args: argparse.Namespace) -> None:
    for source in args.sources:
        record = store.pull(source, **_options(args))
        status = store.status(record)
        print(f"pulled {_label(record)} ({_size(status.size)})")


def _rm(args: argparse.Namespace) -> None:
    for source in args.sources:
        for record in store.remove(source):
            print(f"removed {_label(record)}")


def _run(args: argparse.Namespace) -> None:
    import torch

    from .. import GSPipeline, io

    options = _options(args)
    record = store.resolve(args.source, **options)
    io.image_paths(args.images)  # fail on a bad folder before loading a model
    if args.cameras is None and _camera_input(record) == "required":
        raise ValueError(_needs_cameras(args.source))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    hub_kwargs = {key: value for key, value in options.items() if value is not None}
    pipe = GSPipeline.from_pretrained(args.source, device=device, **hub_kwargs)
    if args.cameras is None and pipe.processor.camera_input == "required":
        raise ValueError(_needs_cameras(args.source))
    views = _read_views(args.images, args.cameras, args.num_views or DEFAULT_NUM_VIEWS)
    out = pipe(views)
    if views.has_poses:
        out = out.to_world()

    args.out.mkdir(parents=True, exist_ok=True)
    ply = out.save_ply(args.out / "scene.ply")
    print(f"{out.gaussians.num_gaussians} Gaussians ({out.space} frame) -> {ply}")
    if out.cameras is not None:  # the cameras the model predicts, as they are
        h, w = out.cameras.image_shape
        k, c2w = out.cameras.normalized_k[0].cpu(), out.cameras.c2w[0].cpu()
        cameras = [io.Camera(str(v), k[v], c2w[v], (w, h)) for v in range(len(c2w))]
        saved = io.save_cameras(args.out / "cameras.json", cameras)
        print(f"{len(cameras)} predicted cameras ({out.space} frame) -> {saved}")


# --- helpers ------------------------------------------------------------------------


def _options(args: argparse.Namespace) -> dict[str, Any]:
    return {"revision": args.revision, "subfolder": args.subfolder}


def _read_views(images: Path, cameras: Path | None, num_views: int):
    """`num_views` evenly spaced images of the folder `images`, all the same size,
    as `Views`; posed by `cameras` (a camera file) if given."""
    import torch

    from .. import Views, io

    paths = io.image_paths(images)
    chosen = None
    if cameras is not None:
        chosen = io.pair_cameras(paths, io.read_cameras(cameras))
    indices = io.evenly_spaced(len(paths), num_views)
    decoded = [io.read_image(paths[i]) for i in indices]
    if len({tuple(image.shape) for image in decoded}) > 1:
        raise ValueError(f"the images of {images} are not all the same size")
    stacked = torch.stack(decoded)
    if chosen is None:
        return Views(stacked)
    chosen = [chosen[i] for i in indices]
    h, w = stacked.shape[-2:]
    for i, camera in zip(indices, chosen):
        cw, ch = camera.size
        if abs(cw / ch - w / h) > ASPECT_TOLERANCE * (w / h):
            raise ValueError(
                f"{paths[i].name} is {w}x{h}, but its camera {camera.name} in "
                f"{cameras} is {cw}x{ch}: a different aspect ratio"
            )
    k = torch.stack([camera.k for camera in chosen])
    c2w = torch.stack([camera.c2w for camera in chosen])
    return Views(stacked, k, c2w, normalized_intrinsics=True)


def _camera_input(record: store.Record) -> str | None:
    """`Processor.camera_input` of a zoo entry or a local directory, read before
    loading it (None when that needs the model: a Hub repo, or a model whose code
    is not installed)."""
    from ..registry import get_model_spec

    try:
        if record.type == "zoo":
            model_type = zoo.get_entry(record.source).model_type
        elif record.type == "local":
            directory = store._local_dir(record)
            model_type = json.loads((directory / "config.json").read_text())[
                "model_type"
            ]
        else:
            return None
        return get_model_spec(model_type).processor_cls.camera_input
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return None


def _needs_cameras(source: str) -> str:
    return (
        f"{source} needs the cameras of its input images: give --cameras "
        "(a transforms.json, a COLMAP model directory or an ffgs cameras.json)"
    )


def _record_rows() -> list[dict[str, Any]]:
    rows = []
    for record in store.records():
        status = store.status(record)
        rows.append(
            {
                "source": record.source,
                "type": record.type,
                "revision": record.revision,
                "subfolder": record.subfolder,
                "commit": record.commit,
                "status": status.state,
                "size": status.size,
                "added": record.added,
                "last_used": record.last_used,
            }
        )
    return rows


def _zoo_rows() -> list[dict[str, Any]]:
    from ..registry import get_model_spec

    recorded = {r.source for r in store.records() if r.type == "zoo"}
    rows = []
    for name in zoo.list_models():
        entry = zoo.get_entry(name)
        try:
            spec = get_model_spec(entry.model_type)
            model = spec.display_name or entry.model_type
            camera_input = spec.processor_cls.camera_input
        except (KeyError, ImportError, ValueError):
            model, camera_input = entry.model_type, None
        status = store.status(store.Record(name, "zoo"))
        rows.append(
            {
                "name": name,
                "model": model,
                "camera_input": camera_input,
                "recorded": name in recorded,
                "status": status.state,
                "size": status.size,
                "description": entry.description,
                "license": entry.license,
            }
        )
    return rows


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    def cell(row: dict[str, Any], key: str) -> str:
        value = row[key]
        if key == "size":
            return _size(value) if value else "-"
        if key == "revision":
            commit = row.get("commit")
            text = value or "main"
            if commit and commit != value:
                text += f" ({commit[:7]})"
            if row.get("subfolder"):
                text += f" /{row['subfolder']}"
            return text if row["type"] == "hub" else "-"
        if key == "recorded":
            return "yes" if value else "-"
        if key == "license":
            return value if len(value) <= 24 else value[:23] + "…"
        if key == "last_used":
            return value[:16].replace("T", " ") if value else "-"
        return "-" if value is None else str(value)

    table = [[title for title, _ in columns]]
    table += [[cell(row, key) for _, key in columns] for row in rows]
    widths = [max(len(line[i]) for line in table) for i in range(len(columns))]
    for line in table:
        print("  ".join(text.ljust(w) for text, w in zip(line, widths)).rstrip())


def _label(record: store.Record) -> str:
    text = record.source
    if record.revision:
        text += f"@{record.revision}"
    if record.subfolder:
        text += f" /{record.subfolder}"
    return text


def _size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    raise AssertionError("unreachable")
