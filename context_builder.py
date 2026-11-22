#!/usr/bin/env python3
"""
Stage S2 — Repo Snapshot & Context Builder

从 base_commit 构建 Repo Snapshot，并为每个 Issue 生成模型实际可见的代码上下文。

核心原则（防泄漏）：
- 仅使用 base_commit 状态的代码，绝不包含修复后的代码
- 不把 Gold Patch、修复后的文件、PR 描述中的答案放入 Context
- Gold Patch 只用于生成标签和监督答案，不用于定位输入构建
- 所有 Context 保存 Hash，保证 E1/E2 使用完全相同的 Context

输出：
  <output_dir>/contexts/<instance_id>.json       上下文快照
  <output_dir>/contexts/<instance_id>.provenance.json 来源与审计元信息
  <output_dir>/contexts/summary.json             汇总统计
"""

import ast
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# 复用 mini_eval_v13 的基础设施（仅用于路径常量，不依赖其 git 克隆）
sys.path.insert(0, r'F:\test\gemma4\harness_v1_20261119')
import mini_eval_v13 as me  # noqa: E402


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中英文混合按字符数/3.5，保守估计）。"""
    return max(1, len(text) // 3)


def posix_path(p: Path | str) -> str:
    """统一路径为正斜杠（POSIX 风格），便于跨平台和 JSON 序列化。"""
    return str(Path(p)).replace("\\", "/")


# ---------------------------------------------------------------------------
# Python 符号提取（AST 基础）
# ---------------------------------------------------------------------------

class SymbolExtractor(ast.NodeVisitor):
    """从 Python 源码中提取符号定义（函数、类、变量、导入）。"""

    def __init__(self, filepath: str, source: str):
        self.filepath = filepath
        self.source = source
        self.symbols: list[dict] = []
        self.imports: list[dict] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.symbols.append({
            "kind": "function",
            "name": node.name,
            "lineno": node.lineno,
            "end_lineno": getattr(node, "end_lineno", node.lineno),
            "file": self.filepath,
        })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.symbols.append({
            "kind": "async_function",
            "name": node.name,
            "lineno": node.lineno,
            "end_lineno": getattr(node, "end_lineno", node.lineno),
            "file": self.filepath,
        })
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbols.append({
            "kind": "class",
            "name": node.name,
            "lineno": node.lineno,
            "end_lineno": getattr(node, "end_lineno", node.lineno),
            "file": self.filepath,
        })
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append({
                "kind": "import",
                "name": alias.name,
                "asname": alias.asname,
                "lineno": node.lineno,
                "file": self.filepath,
            })

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.imports.append({
                "kind": "from_import",
                "module": node.module,
                "name": alias.name,
                "asname": alias.asname,
                "lineno": node.lineno,
                "file": self.filepath,
            })


def extract_symbols_from_file(filepath: Path) -> tuple[list[dict], list[dict]]:
    """解析单个 Python 文件，提取符号和导入。"""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
        extractor = SymbolExtractor(str(filepath), source)
        extractor.visit(tree)
        return extractor.symbols, extractor.imports
    except SyntaxError:
        return [], []
    except Exception:
        return [], []


# ---------------------------------------------------------------------------
# 从 Issue / test_diff 提取符号
# ---------------------------------------------------------------------------

ISSUE_SYMBOL_PATTERNS = [
    # 函数调用：func_name(
    re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('),
    # 类名：ClassName
    re.compile(r'\b([A-Z][a-zA-Z0-9_]*)\b'),
    # 属性访问：obj.method(
    re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\('),
]

STOP_WORDS = {
    'the', 'and', 'for', 'with', 'from', 'import', 'return',
    'if', 'else', 'while', 'for', 'in', 'is', 'not', 'or',
    'True', 'False', 'None', 'self', 'cls', 'Fix', 'fix',
    'bug', 'issue', 'test', 'tests', 'code', 'file', 'files',
}


def extract_symbols_from_text(text: str) -> list[str]:
    """从任意文本中粗略提取候选代码符号。"""
    symbols = set()
    for pattern in ISSUE_SYMBOL_PATTERNS:
        for m in pattern.finditer(text):
            sym = m.group(1)
            if sym.lower() in STOP_WORDS:
                continue
            symbols.add(sym)
    return sorted(symbols)


def extract_imports_from_test_diff(test_diff: str) -> list[str]:
    """从 test_diff 中提取导入的模块名（如 fastapi.responses）。"""
    imports = []
    for m in re.finditer(r'^\+\s*(?:import|from)\s+([a-zA-Z0-9_.]+)', test_diff, re.MULTILINE):
        imports.append(m.group(1))
    return imports


def extract_test_funcs_from_test_diff(test_diff: str) -> list[str]:
    """从 test_diff 中提取测试函数名。"""
    funcs = []
    for m in re.finditer(r'^\+\s*def\s+(test_[a-zA-Z0-9_]*)\s*\(', test_diff, re.MULTILINE):
        funcs.append(m.group(1))
    return funcs


def build_issue_from_test_diff(test_diff: str, instance_id: str) -> str:
    """当 problem_statement 为空时，从 test_diff 构造有意义的 issue 描述。"""
    imports = extract_imports_from_test_diff(test_diff)
    test_funcs = extract_test_funcs_from_test_diff(test_diff)

    parts = [f"Fix the bug described in test changes for {instance_id}."]
    if imports:
        parts.append(f"Relevant imports: {', '.join(imports[:10])}.")
    if test_funcs:
        parts.append(f"Test functions: {', '.join(test_funcs[:10])}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 文件查找与上下文构建
# ---------------------------------------------------------------------------

def _is_excluded(path: Path) -> bool:
    """排除不应纳入训练上下文的文件。"""
    parts = path.parts
    excluded_dirs = {
        ".git", "__pycache__", ".pytest_cache", ".venv", "venv",
        "node_modules", ".github", "build", "dist", "egg-info",
    }
    excluded_names = {
        "conftest.py", "setup.py", "setup.cfg", "pyproject.toml",
        "requirements.txt", "requirements-dev.txt",
    }
    if any(part in excluded_dirs for part in parts):
        return True
    if path.name in excluded_names:
        return True
    return False


def find_files_containing_symbol(repo_dir: Path, symbol: str, max_files: int = 5) -> list[Path]:
    """在 repo 文件中查找包含某个符号的文件（基于文本搜索，保守）。"""
    matches = []
    for p in repo_dir.rglob("*.py"):
        if not p.is_file() or _is_excluded(p):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if symbol in text:
            matches.append(p)
            if len(matches) >= max_files:
                break
    return matches


def read_file_slice(filepath: Path, start_line: int | None = None,
                    end_line: int | None = None, max_lines: int = 80) -> str:
    """读取文件的指定行范围，带行号。"""
    try:
        lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return f"# {filepath}\n# (read error)\n"

    if start_line is None:
        start_line = 1
    if end_line is None:
        end_line = min(start_line + max_lines - 1, len(lines))

    start_line = max(1, start_line)
    end_line = min(len(lines), end_line)

    out = [f"# {posix_path(filepath)} (lines {start_line}-{end_line})"]
    for i in range(start_line - 1, end_line):
        out.append(f"{i+1:4d}: {lines[i]}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Context 预算管理
# ---------------------------------------------------------------------------

class ContextBudget:
    """上下文 token 预算管理。"""

    def __init__(self, max_tokens: int = 12000):
        self.max_tokens = max_tokens
        self.used = 0
        self.included_files: list[str] = []

    def can_add(self, text: str) -> bool:
        cost = estimate_tokens(text)
        return self.used + cost <= self.max_tokens

    def add(self, text: str) -> bool:
        if not self.can_add(text):
            return False
        self.used += estimate_tokens(text)
        return True


# ---------------------------------------------------------------------------
# Repo 文件索引
# ---------------------------------------------------------------------------

def build_repo_file_index(repo_dir: Path) -> dict[str, dict]:
    """构建 repo 中所有 Python 文件的符号索引。"""
    index: dict[str, dict] = {}
    for p in repo_dir.rglob("*.py"):
        if _is_excluded(p):
            continue
        rel = posix_path(p.relative_to(repo_dir))
        symbols, imports = extract_symbols_from_file(p)
        try:
            size_lines = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
        except Exception:
            size_lines = 0
        index[rel] = {
            "symbols": symbols,
            "imports": imports,
            "size_lines": size_lines,
        }
    return index


def verify_repo_commit(repo_dir: Path, base_commit: str) -> tuple[bool, str | None]:
    """验证 repo 当前 HEAD 是否等于 base_commit。

    返回 (是否匹配, 当前 HEAD 或错误信息)。
    """
    if not repo_dir.exists() or not (repo_dir / ".git").exists():
        return False, "repo_dir not a git repository"
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        if head == base_commit:
            return True, head
        # 尝试 checkout base_commit
        try:
            subprocess.check_call(
                ["git", "-C", str(repo_dir), "checkout", base_commit],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            head = subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            return (head == base_commit), head
        except Exception:
            return False, head
    except Exception as e:
        return False, f"git rev-parse failed: {e}"


# ---------------------------------------------------------------------------
# Context 构建主函数
# ---------------------------------------------------------------------------

def build_context(
    instance_id: str,
    issue: str,
    repo_dir: Path,
    base_commit: str = "",
    src_files: list[str] | None = None,
    test_files: list[str] | None = None,
    max_tokens: int = 12000,
    include_repo_tree: bool = True,
) -> dict[str, Any]:
    """
    为单个 Issue 构建代码上下文。

    返回包含上下文内容和 provenance 的 dict。
    """
    context = {
        "instance_id": instance_id,
        "repo_dir": posix_path(repo_dir),
        "issue": issue,
        "base_commit": base_commit,
        "issue_symbols": [],
        "file_tree": [],
        "relevant_files": [],
        "symbol_definitions": {},
        "imports": {},
        "test_files_content": {},
        "file_contents": {},
        "context_budget_tokens": max_tokens,
        "context_used_tokens": 0,
        "context_hash": "",
        "leak_check": {
            "uses_base_commit_only": False,
            "gold_patch_excluded": True,
            "merge_sha_excluded": True,
            "commit_verified": False,
        },
    }

    budget = ContextBudget(max_tokens=max_tokens)

    # P0-2: 实际验证 repo 处于 base_commit，不再�编码 True
    commit_verified, commit_info = verify_repo_commit(repo_dir, base_commit)
    context["leak_check"]["uses_base_commit_only"] = commit_verified
    context["leak_check"]["commit_verified"] = commit_verified
    context["leak_check"]["head"] = commit_info

    # 1. 从 Issue 提取候选符号（不使用 test_diff，防止测试答案泄漏）
    issue_symbols = extract_symbols_from_text(issue)
    # 去重保序
    seen = set()
    uniq_symbols = []
    for s in issue_symbols:
        if s not in seen:
            seen.add(s)
            uniq_symbols.append(s)
    context["issue_symbols"] = uniq_symbols[:50]

    # 2. 构建文件树（轻量，仅目录结构）
    if include_repo_tree and repo_dir.exists():
        tree_lines = []
        for p in sorted(repo_dir.rglob("*")):
            if _is_excluded(p):
                continue
            rel = p.relative_to(repo_dir)
            depth = len(rel.parts)
            if depth > 3:
                continue
            prefix = "  " * (depth - 1) + ("├─ " if p.is_file() else "└─ ")
            tree_lines.append(f"{prefix}{rel.name}")
        tree_text = "\n".join(tree_lines[:200])
        if budget.can_add(tree_text):
            budget.add(tree_text)
            context["file_tree"] = tree_lines[:200]

    # 3. 构建文件符号索引
    file_index: dict[str, dict] = {}
    if repo_dir.exists():
        file_index = build_repo_file_index(repo_dir)

    # 4. 确定相关文件：优先 src_files，其次 issue/test_diff 符号匹配的文件
    relevant_files: list[str] = []
    if src_files:
        for f in src_files:
            pf = Path(f)
            rel = posix_path(pf)
            p = repo_dir / pf
            if p.exists() and not _is_excluded(p):
                relevant_files.append(rel)

    # 通过 issue/test_diff 符号在 repo 中查找额外相关文件
    for sym in context["issue_symbols"][:30]:
        matches = find_files_containing_symbol(repo_dir, sym, max_files=3)
        for m in matches:
            rel = posix_path(m.relative_to(repo_dir))
            if rel not in relevant_files:
                relevant_files.append(rel)

    # 限制相关文件数量
    relevant_files = relevant_files[:30]
    context["relevant_files"] = relevant_files

    # 5. 读取相关文件内容（受预算限制）
    file_contents: dict[str, str] = {}
    symbol_defs: dict[str, list[dict]] = {}
    file_imports: dict[str, list[dict]] = {}

    for rel in relevant_files:
        p = repo_dir / rel
        if not p.exists() or _is_excluded(p):
            continue

        # 提取符号和导入
        symbols, imports = extract_symbols_from_file(p)
        symbol_defs[rel] = symbols
        file_imports[rel] = imports

        # 读取文件内容（全部或受预算限制）
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""

        if budget.can_add(text):
            budget.add(text)
            file_contents[rel] = text
        else:
            # 预算不足，读取部分
            remaining = budget.max_tokens - budget.used
            if remaining > 200:
                slice_text = text[: remaining * 3]
                if budget.can_add(slice_text):
                    budget.add(slice_text)
                    file_contents[rel] = f"# {rel} (truncated due to budget)\n{slice_text}"

    context["file_contents"] = file_contents
    context["symbol_definitions"] = symbol_defs
    context["imports"] = file_imports

    # 6. 测试文件内容（仅列出文件名和关键部分，不泄漏修复逻辑）
    if test_files and repo_dir.exists():
        test_contents: dict[str, str] = {}
        for tf in test_files[:10]:
            p = repo_dir / tf
            if p.exists() and not _is_excluded(p):
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    text = ""
                if budget.can_add(text):
                    budget.add(text)
                    test_contents[tf] = text
        context["test_files_content"] = test_contents

    context["context_used_tokens"] = budget.used

    # 7. 计算 Context Hash（基于 issue + 所有上下文内容，不包含 test_diff 防止泄漏）
    hash_input = json.dumps({
        "instance_id": instance_id,
        "issue": issue,
        "file_tree": context["file_tree"],
        "relevant_files": context["relevant_files"],
        "file_contents": context["file_contents"],
        "symbol_definitions": context["symbol_definitions"],
        "imports": context["imports"],
        "test_files_content": context["test_files_content"],
    }, sort_keys=True, ensure_ascii=False)
    context["context_hash"] = sha256_text(hash_input)

    return context


def save_context(context: dict, output_dir: Path) -> dict[str, Any]:
    """保存 context 并返回 provenance 信息。"""
    instance_id = context["instance_id"]
    ctx_dir = output_dir / "contexts"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    ctx_path = ctx_dir / f"{instance_id}.json"
    prov_path = ctx_dir / f"{instance_id}.provenance.json"

    with open(ctx_path, "w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=1)

    provenance = {
        "instance_id": instance_id,
        "context_file": str(ctx_path),
        "context_hash": context["context_hash"],
        "used_tokens": context["context_used_tokens"],
        "budget_tokens": context["context_budget_tokens"],
        "relevant_files": context["relevant_files"],
        "issue_symbols": context["issue_symbols"],
        "leak_check": context["leak_check"],
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "note": "Context built from base_commit only. Gold patch excluded.",
    }
    prov_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "instance_id": instance_id,
        "context_file": str(ctx_path),
        "provenance_file": str(prov_path),
        "context_hash": context["context_hash"],
        "used_tokens": context["context_used_tokens"],
        "relevant_files_count": len(context["relevant_files"]),
        "issue_symbols_count": len(context["issue_symbols"]),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_contexts_for_tasks(
    task_ids: list[str],
    output_dir: Path,
    max_tokens: int = 12000,
    force_rebuild: bool = False,
) -> list[dict[str, Any]]:
    """为多个任务构建上下文。"""
    recon_path = Path(r'F:\test\gemma4\mini_eval\reconstructed.json')
    recon = {}
    if recon_path.exists():
        recon = json.loads(recon_path.read_text(encoding="utf-8"))

    results = []
    for tid in task_ids:
        task = recon.get(tid, {})
        if not task:
            print(f"[SKIP] {tid}: not found in reconstructed.json")
            continue

        base_commit = task.get("base_commit", "")
        src_files = task.get("src_files", [])
        test_files = task.get("test_files", [])
        test_diff = task.get("test_diff", "")
        problem = task.get("problem_statement", "")

        # 构造有意义的 issue 描述
        if problem and problem.strip():
            issue = problem
        else:
            issue = build_issue_from_test_diff(test_diff, tid)

        # 检查是否已有 context 且不强制重建
        ctx_dir = output_dir / "contexts"
        existing = ctx_dir / f"{tid}.json"
        if existing.exists() and not force_rebuild:
            print(f"[SKIP] {tid}: context already exists")
            continue

        print(f"[BUILD] {tid}: base_commit={base_commit[:12] if base_commit else 'N/A'}")

        repo_dir = Path(r'F:\test\gemma4\mini_eval\repos') / tid
        if not repo_dir.exists():
            print(f"[WARN] {tid}: repo directory not found at {repo_dir}")
            # 仍然生成基于 issue 的上下文（无代码）
            context = build_context(
                instance_id=tid,
                issue=issue,
                repo_dir=repo_dir,
                base_commit=base_commit,
                src_files=src_files,
                test_files=test_files,
                max_tokens=max_tokens,
            )
        else:
            context = build_context(
                instance_id=tid,
                issue=issue,
                repo_dir=repo_dir,
                base_commit=base_commit,
                src_files=src_files,
                test_files=test_files,
                max_tokens=max_tokens,
            )

        res = save_context(context, output_dir)
        results.append(res)
        print(f"[DONE] {tid}: hash={context['context_hash'][:12]} "
              f"tokens={context['context_used_tokens']} "
              f"symbols={len(context['issue_symbols'])} "
              f"files={len(context['relevant_files'])}")

    return results


def main():
    ap = __import__("argparse").ArgumentParser(description="Stage S2: Repo Snapshot & Context Builder")
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="要构建上下文的 task id 列表")
    ap.add_argument("--output-dir", default=r"F:\test\gemma4\training\data\processed",
                    help="处理后数据输出目录")
    ap.add_argument("--max-tokens", type=int, default=12000,
                    help="每个上下文的最大 token 预算")
    ap.add_argument("--force-rebuild", action="store_true",
                    help="强制重建已存在的 context")
    args = ap.parse_args()

    results = build_contexts_for_tasks(
        task_ids=args.tasks,
        output_dir=Path(args.output_dir),
        max_tokens=args.max_tokens,
        force_rebuild=args.force_rebuild,
    )

    # 汇总
    summary_path = Path(args.output_dir) / "contexts" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"results": results, "count": len(results)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nContext builder summary: {len(results)} contexts built")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
