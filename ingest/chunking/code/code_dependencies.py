"""Import / include extraction for dependency metadata.

Author: deviprasad
"""
from __future__ import annotations

import ast
import re
from typing import Iterable, List


def _format_dependencies_field(modules: Iterable[str]) -> str:
    unique = sorted({m.strip() for m in modules if m and str(m).strip()})
    if not unique:
        return ""
    return ", ".join(unique)


def extract_dependencies(content: str, ext: str) -> str:
    """Extract import-like symbols for metadata (comma-separated)."""
    ext_l = ext.lower()
    mods: List[str] = []

    if ext_l == ".py":
        try:
            tree = ast.parse(content)
        except SyntaxError:
            for m in re.finditer(
                r"(?m)^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.,\s]+))\s*",
                content,
            ):
                g1, g2 = m.group(1), m.group(2)
                if g1:
                    mods.append(g1.split(".")[0])
                if g2:
                    for part in g2.replace(",", " ").split():
                        if part and part not in ("import", "as"):
                            mods.append(part.split(".")[0])
        else:

            class V(ast.NodeVisitor):
                def visit_Import(self, node: ast.Import) -> None:
                    for alias in node.names:
                        mods.append((alias.name or "").split(".")[0])

                def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                    if node.module:
                        mods.append(node.module.split(".")[0])
                    for alias in node.names:
                        if alias.name != "*":
                            mods.append(alias.name)

            V().visit(tree)

    elif ext_l in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"):
        for m in re.finditer(
            r"""import\s+(?:[\w*{}\s,]+\s+from\s+)?['"]([^'"]+)['"]|"""
            r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)|"""
            r"""import\s*\(\s*['"]([^'"]+)['"]\s*\)""",
            content,
        ):
            for g in m.groups():
                if g:
                    mods.append(g.strip().split("/")[-1].split(".")[0])

    elif ext_l in (".c", ".h", ".cpp", ".cxx", ".cc", ".hpp", ".hxx"):
        for m in re.finditer(r'#\s*include\s+([<"])([^>"]+)([>"])', content):
            mods.append(m.group(2).strip())

    elif ext_l == ".java":
        for m in re.finditer(r"(?m)^\s*import\s+([\w.]+)\s*;", content):
            mods.append(m.group(1))

    elif ext_l == ".go":
        for m in re.finditer(r'import\s+(?:\(\s*([^)]+)\s*\)|"([^"]+)")', content, re.DOTALL):
            block = m.group(1) or m.group(2) or ""
            for q in re.findall(r'"([^"]+)"', block):
                mods.append(q)
        for m in re.finditer(r'import\s+"([^"]+)"', content):
            mods.append(m.group(1))

    elif ext_l == ".rs":
        for m in re.finditer(r"(?m)^\s*(?:pub\s+)?use\s+([^;]+);", content):
            for seg in m.group(1).split(","):
                seg = seg.split("::")[0].strip()
                if seg and seg not in ("self", "super", "crate"):
                    mods.append(seg)
        for m in re.finditer(r"(?m)^\s*(?:pub\s+)?mod\s+(\w+)\s*;", content):
            mods.append(m.group(1))

    return _format_dependencies_field(mods)
