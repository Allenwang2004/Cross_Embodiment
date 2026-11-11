#!/usr/bin/env python3
"""import_check.py -- static import check for the OOP refactor (docs/refactor/PLAN.md, step 1 on).

Every .py file under the scanned folders (default: scripts, model, tests, utils and, once it exists, the crossenbodiment
package) must parse, and every import of a REPO-LOCAL module in it must resolve to a file that exists, anywhere in the
file (imports inside functions count). Nothing is imported or run: the files are read with ast.

Resolution follows how the code runs: `python scripts/x.py` puts scripts/ on sys.path, and the scripts insert the
repository root and scripts/ themselves, so a module name is looked up (1) next to the importing file, (2) at the
repository root, (3) in scripts/. Relative imports resolve against the importing file's package; a folder without
__init__.py counts as a namespace package (scripts/). importlib.import_module("name") / __import__("name") with a string
literal count as imports.

A name is repo-local when it resolves to a repo file, or when its top-level name is a top-level package of the repository
or the stem of any .py file in it (archive/ included, so `import controlled_pairs` from a live script is caught after
controlled_pairs.py moved to archive/scripts/). Everything else must be importable from the environment
(importlib.util.find_spec of the top-level name, nothing executed); a missing one is a warning (Blender's bpy, optional
wandb / pytest), not a failure.

Names are checked too, so later steps that move functions are caught:
  * `from M import a`: a is defined at M's top level (def, class, assignment, import, or inside a top-level
    if / try / with / for) or is a submodule of the package M. A module with `from ... import *` or a module-level
    __getattr__ accepts any name.
  * `import M as A` / `import M`: every LOADED attribute A.x (ast.Load; assignments like `P.OUT = OUT` are skipped)
    must be defined in M the same way. Attributes reached through dotted names (`model.losses.d_root`) are checked
    on the last module of the chain.

usage: uv run python tests/golden/import_check.py [--dirs scripts model ...] [--quiet]
exit 0 = every file parses and every repo-local import / name resolves; 1 otherwise.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DIRS = ("scripts", "model", "tests", "utils", "crossenbodiment")
SKIP_PARTS = {"__pycache__", ".venv", ".git"}


def py_files(root: Path):
    for p in sorted(root.rglob("*.py")):
        if not SKIP_PARTS.intersection(p.relative_to(REPO).parts):
            yield p


class Module:
    """Top-level names of one repo file (parsed lazily, cached)."""
    cache: dict[Path, "Module"] = {}

    def __init__(self, path: Path):
        self.path = path
        self.names: set[str] = set()
        self.accept_any = False
        if path.is_dir():                         # namespace package: only submodules
            return
        tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
        self._collect(tree.body)

    @classmethod
    def get(cls, path: Path) -> "Module":
        if path not in cls.cache:
            cls.cache[path] = Module(path)
        return cls.cache[path]

    def _target_names(self, t):
        if isinstance(t, ast.Name):
            self.names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._target_names(e)
        elif isinstance(t, ast.Starred):
            self._target_names(t.value)

    def _collect(self, body):
        for s in body:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.names.add(s.name)
                if s.name == "__getattr__" and not isinstance(s, ast.ClassDef):
                    self.accept_any = True
            elif isinstance(s, ast.Assign):
                for t in s.targets:
                    self._target_names(t)
            elif isinstance(s, (ast.AnnAssign, ast.AugAssign)):
                self._target_names(s.target)
            elif isinstance(s, ast.Import):
                for a in s.names:
                    self.names.add(a.asname or a.name.split(".")[0])
            elif isinstance(s, ast.ImportFrom):
                for a in s.names:
                    if a.name == "*":
                        self.accept_any = True
                    else:
                        self.names.add(a.asname or a.name)
            elif isinstance(s, ast.If):
                self._collect(s.body); self._collect(s.orelse)
            elif isinstance(s, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                self._collect(s.body); self._collect(s.orelse); self._collect(s.finalbody)
                for h in s.handlers:
                    self._collect(h.body)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                for it in s.items:
                    if it.optional_vars is not None:
                        self._target_names(it.optional_vars)
                self._collect(s.body)
            elif isinstance(s, (ast.For, ast.AsyncFor, ast.While)):
                if isinstance(s, (ast.For, ast.AsyncFor)):
                    self._target_names(s.target)
                self._collect(s.body); self._collect(s.orelse)
            elif isinstance(s, ast.Global):
                self.names.update(s.names)
            elif isinstance(s, ast.Expr) and isinstance(s.value, ast.NamedExpr):
                self._target_names(s.value.target)
        # names declared `global` inside functions and assigned there are module attributes too
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Global):
                self.names.update(node.names)

    def has(self, name: str) -> bool:
        if self.accept_any or name in self.names:
            return True
        return submodule(self.path, name) is not None


def submodule(pkg: Path, name: str):
    """pkg is a package file (__init__.py) or a namespace dir: its submodule `name`, else None."""
    base = pkg if pkg.is_dir() else (pkg.parent if pkg.name == "__init__.py" else None)
    if base is None:
        return None
    for c in (base / f"{name}.py", base / name / "__init__.py"):
        if c.is_file():
            return c
    if (base / name).is_dir() and any((base / name).glob("*.py")):
        return base / name
    return None


def resolve_parts(parts, roots):
    """The repo path of the dotted module `parts` (a .py file, a package __init__.py or a namespace dir), else None."""
    for root in roots:
        cur = root
        for i, p in enumerate(parts):
            last = i == len(parts) - 1
            if last and (cur / f"{p}.py").is_file():
                return cur / f"{p}.py"
            pkg = cur / p
            if (pkg / "__init__.py").is_file():
                if last:
                    return pkg / "__init__.py"
            elif not (pkg.is_dir() and any(pkg.glob("*.py"))):
                break                             # not under this root: try the next one
            elif last:
                return pkg                        # namespace package
            cur = pkg
    return None


def local_stems():
    stems, tops = set(), set()
    for p in py_files(REPO):
        rel = p.relative_to(REPO)
        stems.add(p.stem)
        if len(rel.parts) > 1:
            tops.add(rel.parts[0])
    tops.discard("archive")
    return stems, tops


class Checker:
    def __init__(self):
        self.stems, self.tops = local_stems()
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.n_local = self.n_names = self.n_ext = 0
        self._ext_cache: dict[str, bool] = {}

    def roots_for(self, f: Path):
        return [f.parent, REPO, REPO / "scripts"]

    def is_local_name(self, top: str) -> bool:
        return top in self.tops or top in self.stems

    def external_ok(self, top: str) -> bool:
        if top not in self._ext_cache:
            if top in sys.stdlib_module_names or top in sys.builtin_module_names:
                self._ext_cache[top] = True
            else:
                try:
                    self._ext_cache[top] = importlib.util.find_spec(top) is not None
                except (ImportError, ValueError):
                    self._ext_cache[top] = False
        return self._ext_cache[top]

    def module(self, f: Path, dotted: str, lineno: int, *, level: int = 0):
        """Resolve one imported module; record an error / warning; return its repo path or None."""
        rel = f.relative_to(REPO)
        if level:
            base = f.parent
            for _ in range(level - 1):
                base = base.parent
            parts = dotted.split(".") if dotted else []
            target = resolve_parts(parts, [base]) if parts else (base / "__init__.py" if (base / "__init__.py").is_file() else base)
            self.n_local += 1
            if target is None:
                self.errors.append(f"{rel}:{lineno}: relative import {'.' * level}{dotted} does not resolve")
            return target
        parts = dotted.split(".")
        target = resolve_parts(parts, self.roots_for(f))
        if target is not None:
            self.n_local += 1
            return target
        if self.is_local_name(parts[0]):
            self.n_local += 1
            self.errors.append(f"{rel}:{lineno}: import {dotted}: repo-local module not found "
                               f"(looked next to the file, at the repo root and in scripts/)")
            return None
        self.n_ext += 1
        if not self.external_ok(parts[0]):
            self.warnings.append(f"{rel}:{lineno}: import {dotted}: not found in the environment")
        return None

    def name(self, f: Path, mod: Path, name: str, lineno: int, how: str):
        self.n_names += 1
        try:
            ok = Module.get(mod).has(name)
        except SyntaxError:
            return                                # reported when that file itself is scanned
        if not ok:
            self.errors.append(f"{f.relative_to(REPO)}:{lineno}: {how}: '{name}' is not defined in "
                               f"{mod.relative_to(REPO)}")

    def check_file(self, f: Path):
        rel = f.relative_to(REPO)
        try:
            tree = ast.parse(f.read_text(errors="replace"), filename=str(rel))
        except SyntaxError as e:
            self.errors.append(f"{rel}:{e.lineno}: does not parse: {e.msg}")
            return
        aliases: dict[str, Path] = {}             # local alias -> repo module file, for A.attr checks
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    target = self.module(f, a.name, node.lineno)
                    if target is None:
                        continue
                    if a.asname:
                        aliases[a.asname] = target
                    else:                         # `import a.b.c` binds `a`; a.b.c.x is checked below via chains
                        top = resolve_parts([a.name.split(".")[0]], self.roots_for(f))
                        if top is not None:
                            aliases.setdefault(a.name.split(".")[0], top)
            elif isinstance(node, ast.ImportFrom):
                target = self.module(f, node.module or "", node.lineno, level=node.level)
                if target is None:
                    continue
                for a in node.names:
                    if a.name == "*":
                        continue
                    self.name(f, target, a.name, node.lineno, f"from {'.' * node.level}{node.module or ''} import")
                    sub = submodule(target, a.name)
                    if sub is not None:
                        aliases[a.asname or a.name] = sub
            elif isinstance(node, ast.Call):
                fn = node.func
                fname = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
                if (fname in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)):
                    self.module(f, node.args[0].value, node.lineno)
        if not aliases:
            return
        # A.x loads (and A.b.x chains through submodules) on a repo-local module alias
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)):
                continue
            chain = []
            v = node
            while isinstance(v, ast.Attribute):
                chain.append(v.attr)
                v = v.value
            if not (isinstance(v, ast.Name) and v.id in aliases):
                continue
            chain.reverse()                       # A.b.c.x -> [b, c, x]
            mod = aliases[v.id]
            for i, attr in enumerate(chain):
                sub = submodule(mod, attr)
                if sub is not None and i < len(chain) - 1:
                    mod = sub
                    continue
                # only the first non-submodule attribute is a name of `mod`; deeper ones are attributes of objects
                self.name(f, mod, attr, node.lineno, f"{v.id}.{'.'.join(chain[:i + 1])}")
                break
            # the inner Attribute nodes of the same chain are visited too; checking them again is harmless


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dirs", nargs="+", default=list(DEFAULT_DIRS))
    ap.add_argument("--quiet", action="store_true", help="do not list the warnings")
    a = ap.parse_args()
    files = [p for d in a.dirs if (REPO / d).is_dir() for p in py_files(REPO / d)]
    c = Checker()
    for f in files:
        c.check_file(f)
    errors = sorted(set(c.errors))
    warnings = sorted(set(c.warnings))
    if warnings and not a.quiet:
        print(f"warnings ({len(warnings)}): external modules not found in this environment")
        for w in warnings:
            print("  " + w)
    for e in errors:
        print("ERROR " + e)
    scanned = ", ".join(d for d in a.dirs if (REPO / d).is_dir())
    print(f"{len(files)} files in {scanned}: {c.n_local} repo-local imports, {c.n_names} imported names / module "
          f"attributes, {c.n_ext} external imports checked; {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
