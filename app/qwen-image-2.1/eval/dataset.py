"""jsonl 评估数据集（文生图 / 图片编辑共用）。

格式与 ``diffusion-svdq`` 的 jsonl 数据集一致，但每条样本必须额外包含
``output`` 字段：相对 ``metadata.jsonl`` 目录的 ground-truth 图片，供 FID
参考分布使用。``image`` 为空表示文生图样本；非空表示图片编辑样本，可以是
单个文件名或文件名列表。
"""

import json
import os
import random
import typing as tp
from dataclasses import dataclass

from PIL import Image

__all__ = [
    "EvalSample",
    "append_manifest_entry",
    "load_eval_samples",
    "manifest_path",
    "open_conditional_images",
    "read_manifest",
]

_METADATA_FILE = "metadata.jsonl"
_MANIFEST_FILE = "manifest.jsonl"
_KNOWN_FIELDS = frozenset({"prompt", "output", "image", "width", "height"})


@dataclass(frozen=True)
class EvalSample:
    """一条评估样本。

    Args:
        index (`int`): ``metadata.jsonl`` 中的物理行号减 1（0 起）。
        sample_id (`str`): ``sample_{index:05d}``，输出图片与指标对齐的唯一键。
        prompt (`str`): 提示词（原样保留）。
        width (`int | None`): 文生图输出宽度；未提供为 `None`。
        height (`int | None`): 文生图输出高度；未提供为 `None`。
        images (`tuple[str, ...]`): 条件输入图片绝对路径；文生图样本为空元组。
        output (`str`): ground-truth 图片绝对路径。
    """

    index: int
    sample_id: str
    prompt: str
    width: int | None
    height: int | None
    images: tuple[str, ...]
    output: str


def _non_empty_str(obj: dict[str, tp.Any], key: str, path: str, lineno: int) -> str:
    value = obj[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{lineno}: `{key}` must be a non-empty string, got {value!r}")
    return value


def _positive_int(obj: dict[str, tp.Any], key: str, path: str, lineno: int) -> int:
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}:{lineno}: `{key}` must be a positive integer, got {value!r}")
    return value


def _image_paths(obj: dict[str, tp.Any], root: str, path: str, lineno: int) -> tuple[str, ...]:
    value = obj["image"]
    if isinstance(value, str):
        names = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        names = value
    else:
        raise ValueError(f"{path}:{lineno}: `image` must be a string or a list of strings, got {value!r}")
    if not names:
        raise ValueError(f"{path}:{lineno}: `image` must not be an empty list")
    paths: list[str] = []
    for name in names:
        if not name.strip():
            raise ValueError(f"{path}:{lineno}: `image` contains an empty file name")
        paths.append(os.path.normpath(os.path.join(root, name)))
    return tuple(paths)


def _parse_sample(obj: tp.Any, lineno: int, path: str, root: str) -> EvalSample:
    if not isinstance(obj, dict):
        raise TypeError(f"{path}:{lineno}: each line must be a JSON object, got {type(obj).__name__}")

    if "prompt" not in obj:
        raise ValueError(f"{path}:{lineno}: missing required field `prompt`")
    prompt = _non_empty_str(obj, "prompt", path, lineno)

    if "output" not in obj:
        raise ValueError(f"{path}:{lineno}: missing required field `output`")
    output = os.path.normpath(os.path.join(root, _non_empty_str(obj, "output", path, lineno)))

    width = _positive_int(obj, "width", path, lineno) if "width" in obj else None
    height = _positive_int(obj, "height", path, lineno) if "height" in obj else None
    images = _image_paths(obj, root, path, lineno) if "image" in obj else ()

    unknown = set(obj) - _KNOWN_FIELDS
    if unknown:
        print(f"{path}:{lineno}: ignoring unknown fields {sorted(unknown)}")

    index = lineno - 1
    return EvalSample(
        index=index,
        sample_id=f"sample_{index:05d}",
        prompt=prompt,
        width=width,
        height=height,
        images=images,
        output=output,
    )


def load_eval_samples(root: str, num_samples: int = -1, seed: int = 0) -> list[EvalSample]:
    """加载评估数据集并随机选取 ``num_samples`` 条样本。

    选择语义与 ``diffusion-svdq`` 一致：``random.Random(seed).shuffle`` → 取前 N
    → 按行号升序返回，因此同一 ``seed`` 的选择结果确定。字段校验对所有行进行，
    图片/ground-truth 文件存在性只对被选中的样本检查。
    """
    path = os.path.join(root, _METADATA_FILE)
    if not os.path.isfile(path):
        raise ValueError(f"metadata file not found: {path!r}")

    samples: list[EvalSample] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            samples.append(_parse_sample(obj, lineno, path, root))

    if not samples:
        raise ValueError(f"no samples found in {path!r}")

    positions = list(range(len(samples)))
    if 0 < num_samples < len(positions):
        random.Random(seed).shuffle(positions)
        positions = sorted(positions[:num_samples])

    selected = [samples[i] for i in positions]
    for sample in selected:
        for image_path in sample.images:
            if not os.path.isfile(image_path):
                raise ValueError(f"{sample.sample_id}: image file not found: {image_path!r}")
        if not os.path.isfile(sample.output):
            raise ValueError(f"{sample.sample_id}: ground-truth file not found: {sample.output!r}")
    return selected


def open_conditional_images(sample: EvalSample) -> list[Image.Image]:
    """按原始顺序打开条件图片；调用方负责关闭。"""
    opened: list[Image.Image] = []
    try:
        for image_path in sample.images:
            image = Image.open(image_path)
            image.load()
            opened.append(image)
    except Exception as exc:
        for image in opened:
            image.close()
        raise ValueError(f"{sample.sample_id}: failed to open image: {exc}") from exc
    return opened


def manifest_path(output_root: str) -> str:
    return os.path.join(output_root, _MANIFEST_FILE)


def read_manifest(path: str) -> dict[str, dict[str, tp.Any]]:
    """读取 manifest；忽略空行与末尾不完整的行，按 ``sample_id`` 去重（后者覆盖前者）。"""
    entries: dict[str, dict[str, tp.Any]] = {}
    if not os.path.isfile(path):
        return entries
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break
            entries[entry["sample_id"]] = entry
    return entries


def append_manifest_entry(path: str, entry: dict[str, tp.Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
