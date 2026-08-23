"""Generate source/ONNX operator inventories with explicit RKNN uncertainty."""

from __future__ import annotations

import ast
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class OperatorRow:
    graph: str
    operator: str
    count: int
    shape: str
    shape_mode: str
    rknn_support: str
    fallback_plan: str
    evidence: str


KNOWN_RKNN = {
    "GridSample": (
        "NOT_SUPPORTED_RKNN_TOOLKIT2_2.3.2_REFERENCE",
        "Keep coordinate sampling on CPU NEON/Mali GPU or a verified custom implementation.",
    ),
    "grid_sample": (
        "NOT_SUPPORTED_RKNN_TOOLKIT2_2.3.2_REFERENCE",
        "Keep coordinate sampling on CPU NEON/Mali GPU or a verified custom implementation.",
    ),
    "Einsum": (
        "NOT_SUPPORTED_RKNN_TOOLKIT2_2.3.2_REFERENCE",
        "Use the numerically tested reshape + batched MatMul candidate.",
    ),
    "einsum": (
        "NOT_SUPPORTED_RKNN_TOOLKIT2_2.3.2_REFERENCE",
        "Use the numerically tested reshape + batched MatMul candidate.",
    ),
}


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def scan_python_operators(paths: Iterable[str | Path]) -> list[OperatorRow]:
    counts: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}
    for path_value in paths:
        path = Path(path_value)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            qualified = _qualified_name(node.func)
            operator = qualified.rsplit(".", 1)[-1]
            if operator not in {
                "grid_sample",
                "einsum",
                "conv2d",
                "avg_pool2d",
                "interpolate",
                "linear",
                "matmul",
                "bmm",
                "softmax",
                "layer_norm",
            }:
                continue
            counts[operator] = counts.get(operator, 0) + 1
            evidence.setdefault(operator, []).append(f"{path}:{getattr(node, 'lineno', '?')}")
    rows = []
    for operator, count in sorted(counts.items()):
        support, fallback = KNOWN_RKNN.get(
            operator,
            (
                "REQUIRES_ACTUAL_TOOLKIT_CONVERSION",
                "Test the fixed-shape subgraph with the installed RKNN Toolkit2 version.",
            ),
        )
        rows.append(
            OperatorRow(
                graph="python_source_call_chain",
                operator=operator,
                count=count,
                shape="source-dependent",
                shape_mode="unknown_until_export",
                rknn_support=support,
                fallback_plan=fallback,
                evidence="; ".join(evidence[operator]),
            )
        )
    return rows


def load_onnx_operators(path: str | Path) -> list[OperatorRow]:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("install the onnx extra to inspect exported graphs") from exc
    model = onnx.shape_inference.infer_shapes(onnx.load(str(path)))
    counts: dict[str, int] = {}
    shapes: dict[str, set[str]] = {}
    shape_modes: dict[str, set[str]] = {}
    value_shapes: dict[str, str] = {}
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dimensions = []
        static = True
        for dimension in tensor_type.shape.dim:
            if dimension.HasField("dim_value"):
                dimensions.append(str(dimension.dim_value))
            elif dimension.HasField("dim_param"):
                dimensions.append(dimension.dim_param)
                static = False
            else:
                dimensions.append("?")
                static = False
        value_shapes[value.name] = "[" + ",".join(dimensions) + "]"
        value_shapes[f"{value.name}::mode"] = "static" if static else "dynamic"
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
        inputs = [value_shapes[name] for name in node.input if name in value_shapes]
        outputs = [value_shapes[name] for name in node.output if name in value_shapes]
        shapes.setdefault(node.op_type, set()).add(
            f"{'|'.join(inputs) or '?'} -> {'|'.join(outputs) or '?'}"
        )
        modes = [
            value_shapes.get(f"{name}::mode", "unknown")
            for name in (*node.input, *node.output)
            if name in value_shapes
        ]
        shape_modes.setdefault(node.op_type, set()).add(
            "dynamic" if "dynamic" in modes else "static" if modes else "unknown"
        )
    rows = []
    for operator, count in sorted(counts.items()):
        support, fallback = KNOWN_RKNN.get(
            operator,
            (
                "REQUIRES_ACTUAL_TOOLKIT_CONVERSION",
                "Verify with the installed converter and record its exact diagnostic.",
            ),
        )
        rows.append(
            OperatorRow(
                graph=Path(path).name,
                operator=operator,
                count=count,
                shape="; ".join(sorted(shapes[operator])[:6]),
                shape_mode="/".join(sorted(shape_modes[operator])),
                rknn_support=support,
                fallback_plan=fallback,
                evidence=str(Path(path).resolve()),
            )
        )
    return rows


def write_operator_csv(path: str | Path, rows: list[OperatorRow]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(asdict(rows[0])), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
