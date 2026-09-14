#!/usr/bin/env python3
"""Compile, validate, and package the anonymous paper and reproducibility files."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
from pypdf import PdfReader
from reproduce import EXPERIMENTS

ROOT = Path(__file__).resolve().parent
LIMIT_BYTES = 20_000_000
TARGET_BYTES = 19_000_000
EXPECTED_TESTS = 39
POLICIES = ("q2", "q3", "q4_2", "q4_3")
RESULTS = ("result1.xlsx", "result2.xlsx", "result3.xlsx", "result4-2.xlsx", "result4-3.xlsx")
TEXT_SUFFIXES = {".py", ".md", ".tex", ".bib", ".json", ".csv", ".txt", ".ini"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def compact_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def validate_experiment_report(report):
    expected = [policy + "_" + variant for policy, variant in EXPERIMENTS]
    checks = report.get("results", {})
    complete = (
        len(expected) == len(set(expected)) == 33
        and report.get("expected_experiments") == 33
        and report.get("verified_experiments") == 33
        and report.get("complete") is True
        and report.get("passed") is True
        and not report.get("pending") and not report.get("errors")
        and set(checks) == set(expected)
        and all(value.get("passed") is True and value.get("intervals") == 365 * 144 for value in checks.values())
    )
    if not complete:
        raise RuntimeError("all 33 full-year experiments must have a complete passing verification report before final packaging")
    return expected


def verify_experiment_inputs():
    report = json.loads((ROOT / "artifacts/experiment_verification.json").read_text(encoding="utf-8"))
    expected = validate_experiment_report(report)
    for name in expected:
        base = ROOT / "artifacts/experiments" / name
        summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
        if summary.get("whole_year", {}).get("days") != 365 or summary.get("evaluation", {}).get("days") != 334:
            raise RuntimeError(name + ": experiment is not a complete 365-day run")
        hashes = report["results"][name].get("file_sha256", {})
        for key, filename in (("ledger", "ledger.npz"), ("summary", "summary.json")):
            if hashes.get(key) != digest(base / filename):
                raise RuntimeError(name + ": experiment changed after its independent verification")
    return report, expected


def source_files():
    """Return only the runnable modelling code shipped as support material."""
    fixed = [ROOT / name for name in ("run.py", "requirements.txt", "README.md")]
    files = fixed + list((ROOT / "src/microgrid").glob("*.py"))
    return sorted(set(files))


def expected_support_paths():
    return ({path.relative_to(ROOT).as_posix() for path in source_files()}
            | set(RESULTS) | {"AI工具使用详情.pdf"})


def private_markers(text):
    # Construct generic home-directory patterns without shipping a personal path.
    prefixes = ["/" + word + "/" for word in ("Users", "home")]
    patterns = [re.escape(prefix) + r"[A-Za-z0-9_.-]+(?:/|\b)" for prefix in prefixes]
    patterns.append(r"[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+")
    found = ["home_directory" for pattern in patterns if re.search(pattern, text)]
    tokens = {Path.home().name, os.environ.get("USER", "")}
    for token in tokens - {"", "root", "user", "runner", "admin", "administrator"}:
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])", text, re.I):
            found.append("local_account_name")
    return sorted(set(found))


def inspect_pdf(path, role="main"):
    reader = PdfReader(path)
    errors, page_text, footers = [], [], []
    if reader.is_encrypted:
        errors.append("PDF must not be encrypted")
    for index, page in enumerate(reader.pages, 1):
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        if abs(width - 595.276) > 1 or abs(height - 841.89) > 1:
            errors.append("page {} is not portrait A4".format(index))
        pieces = []
        def visit(text, cm, tm, font, size):
            stripped = text.strip()
            if not stripped:
                return
            x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            if re.fullmatch(r"\d+", stripped) and 0 <= y <= 100 and abs(x - width / 2) <= 35:
                pieces.append((stripped, x, y))
        text = page.extract_text(visitor_text=visit) or ""
        page_text.append(text)
        matched = [piece for piece in pieces if piece[0] == str(index)]
        if not matched:
            errors.append("page {} lacks its centered continuous Arabic footer".format(index))
        footers.append({"page": index, "matched": bool(matched), "x": round(matched[-1][1], 3) if matched else None, "y": round(matched[-1][2], 3) if matched else None})
    compact = [re.sub(r"\s+", "", text) for text in page_text]
    full = "\n".join(compact)
    metadata = dict(reader.metadata or {})
    author = str(metadata.get("/Author", "")).strip()
    if author:
        errors.append("nonempty PDF author metadata")
    if private_markers(full + json.dumps(metadata, ensure_ascii=False, default=str)):
        errors.append("personal account or home-directory information in PDF")
    result = {"pages": len(reader.pages), "all_pages_a4": not any("A4" in error for error in errors), "continuous_centered_footers": all(item["matched"] for item in footers), "footer_positions": footers, "author_metadata_empty": not author}
    if role == "main":
        if len(compact) < 3:
            errors.append("main paper is too short for the required structure")
        title = re.sub(r"\s+", "", str(metadata.get("/Title", "")))
        if not compact or not title or title not in compact[0] or "摘要" not in compact[0] or "关键词" not in compact[0]:
            errors.append("first page must contain abstract and keywords")
        if len(compact) < 2 or not re.match(r"(?:(?:一|1)[.、]?)?问题重述(?:与分析)?", compact[1]):
            errors.append("second page must begin the first main section")
        def first_page(token):
            return next((i + 1 for i, value in enumerate(compact) if token in value), None)
        ai = first_page("AI工具使用声明")
        reference = first_page("参考文献")
        appendix = next((i + 1 for i, value in enumerate(compact)
                         if reference and i + 1 > reference
                         and ("支撑材料文件说明" in value or "支撑文件目录" in value)), None)
        if ai is None or reference is None or appendix is None:
            errors.append("required AI statement, references, or appendix directory not found")
        elif full.find("AI工具使用声明") >= full.find("参考文献") or not reference < appendix:
            errors.append("AI statement / references / appendix order is wrong")
        if appendix:
            body_pages = appendix - 2  # body begins on page 2; includes references.
            if body_pages > 30:
                errors.append("body including references exceeds 30 pages")
            if not all(filename in compact[appendix - 1] for filename in RESULTS):
                errors.append("appendix first page does not list all five required result files")
        else:
            body_pages = None
        body_text = "\n".join(compact[:appendix - 1]) if appendix else full
        for placeholder in ("待计算", "计算结果生成后载入", "占位符", "PLACEHOLDER", "TBD"):
            if placeholder in body_text:
                errors.append("unresolved placeholder in compiled main PDF")
        source = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
        has_contents = bool(re.search(r"\\tableofcontents\b", source))
        has_header = bool(re.search(r"\\fancyhead(?:\[[^]]*\])?\s*\{[^}]+\}|\\pagestyle\s*\{(?:headings|myheadings)\}", source))
        if has_contents:
            errors.append("a table of contents is present in the source")
        if has_header:
            errors.append("a running header is configured in the source")
        result.update(ai_statement_page=ai, reference_page=reference, appendix_page=appendix, body_pages=body_pages, pre_appendix_pages=appendix - 1 if appendix else None, no_table_of_contents=not has_contents, no_configured_running_header=not has_header)
    elif not compact or "AI工具使用详情" not in compact[0]:
        errors.append("AI document title missing")
    result.update(bytes=Path(path).stat().st_size, sha256=digest(path), passed=not errors, errors=errors)
    if result["bytes"] >= LIMIT_BYTES:
        result["errors"].append("PDF is not smaller than 20,000,000 bytes")
        result["passed"] = False
    return result


def compile_tex(name):
    if not shutil.which("tectonic"):
        raise RuntimeError("tectonic is unavailable; install it or use --no-compile with current PDFs")
    command = ["tectonic", "--keep-logs", "--keep-intermediates", name + ".tex"]
    result = subprocess.run(command, cwd=ROOT / "paper", text=True, capture_output=True, timeout=600)
    log_path = ROOT / "paper" / (name + ".log")
    transcript = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    evidence = result.stdout + "\n" + result.stderr + "\n" + transcript
    fatal = [line for line in evidence.splitlines() if re.search(r"^!|Missing character:|undefined references|Citation .* undefined|^error:", line, re.I)]
    if result.returncode or fatal:
        excerpt = "\n".join(fatal[-20:] or evidence.splitlines()[-30:])
        raise RuntimeError("{} compilation failed or left unresolved content:\n{}".format(name, excerpt))
    overfull = [float(x) for x in re.findall(r"Overfull \\[hv]box \(([0-9.]+)pt", transcript)]
    return {"document": name, "returncode": result.returncode, "overfull_boxes": len(overfull), "max_overfull_pt": max(overfull or [0])}


def run_tests():
    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT,
                            text=True, capture_output=True, timeout=600)
    transcript = result.stdout + result.stderr
    match = re.search(r"(\d+) passed", transcript)
    count = int(match.group(1)) if match else None
    if result.returncode or count != EXPECTED_TESTS:
        raise RuntimeError("automated tests failed or their documented count changed:\n" + transcript[-4000:])
    normalized = "{} passed\n".format(count)
    (ROOT / "artifacts/test_report.txt").write_text(normalized, encoding="utf-8")
    return {"passed": True, "tests": count}


def copy_support(support):
    support.mkdir(parents=True, exist_ok=True)
    for source in source_files():
        if not source.is_file():
            raise RuntimeError("missing source: " + str(source.relative_to(ROOT)))
        target = support / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in RESULTS:
        source = ROOT / "artifacts/exports" / name
        if not source.is_file():
            raise RuntimeError("missing generated workbook: " + name)
        shutil.copy2(source, support / name)


def audit_support(support):
    errors, files = [], sorted(path for path in support.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(support).as_posix()
        if path.is_symlink():
            errors.append(relative + ": symbolic links are not permitted in the package")
        if any(part in {".venv", "__pycache__", ".git", "C题", "node_modules", "__MACOSX"} for part in path.relative_to(support).parts) or path.name == ".DS_Store" or path.suffix in {".zip", ".pyc", ".aux", ".log", ".bbl", ".blg", ".out"}:
            errors.append(relative + ": environment, raw input, archive, or build intermediate is not permitted")
        if path.suffix.lower() in TEXT_SUFFIXES:
            if private_markers(path.read_text(encoding="utf-8", errors="replace")):
                errors.append(relative + ": personal information")
        elif path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                if private_markers(stream.read()):
                    errors.append(relative + ": personal information in compressed data")
        elif path.suffix == ".xlsx":
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if member.endswith(".xml"):
                        content = archive.read(member).decode("utf-8", errors="replace")
                        if private_markers(content):
                            errors.append(relative + ": personal information in workbook XML")
                        if member == "docProps/core.xml" and re.search(r"<(?:dc:creator|cp:lastModifiedBy)>\s*[^<\s]", content):
                            errors.append(relative + ": nonempty workbook author metadata")
        elif path.suffix == ".pdf":
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if private_markers(text + json.dumps(dict(reader.metadata or {}), default=str)):
                errors.append(relative + ": personal information in embedded PDF")
    for name in RESULTS:
        if not (support / name).is_file():
            errors.append("missing required workbook: " + name)
    actual = {path.relative_to(support).as_posix() for path in files}
    expected = expected_support_paths()
    for name in sorted(expected - actual):
        errors.append("missing required support file: " + name)
    for name in sorted(actual - expected):
        errors.append("unexpected support file: " + name)
    return {"passed": not errors, "errors": errors, "files": len(files), "uncompressed_bytes": sum(path.stat().st_size for path in files)}


def make_manifest(support):
    entries = [{"path": path.relative_to(support).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)} for path in sorted(support.rglob("*")) if path.is_file() and path != support / "manifest.json"]
    compact_json(support / "manifest.json", {"hash_algorithm": "sha256", "self_excluded": "manifest.json", "files": entries})
    return entries


def inspect_zip(path):
    errors = []
    with zipfile.ZipFile(path) as archive:
        damaged = archive.testzip()
        if damaged:
            errors.append("ZIP CRC failure: " + damaged)
        expected = expected_support_paths()
        names = archive.namelist()
        if set(names) != expected or len(names) != len(expected):
            errors.append("ZIP contents differ from the minimal support file list")
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                errors.append("unsafe ZIP path")
    size = Path(path).stat().st_size
    if size >= LIMIT_BYTES:
        errors.append("ZIP is not smaller than 20,000,000 bytes")
    return {"passed": not errors, "errors": errors, "bytes": size, "below_19mb_target": size < TARGET_BYTES, "sha256": digest(path), "files": len(expected)}


def compare_file_pairs(pairs):
    """Return human-readable errors when two expected copies are not identical."""
    errors = []
    for label, first, second in pairs:
        first, second = Path(first), Path(second)
        if not first.is_file():
            errors.append(label + ": current file is missing")
        elif not second.is_file():
            errors.append(label + ": packaged copy is missing")
        elif digest(first) != digest(second):
            errors.append(label + ": packaged copy is stale")
    return errors


def audit_manifest_tree(support):
    """Verify that the expanded support directory exactly matches its manifest."""
    support = Path(support)
    manifest_path = support / "manifest.json"
    if not manifest_path.is_file():
        return ["expanded support manifest is missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {item["path"]: item for item in manifest["files"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return ["expanded support manifest is invalid: " + str(exc)]
    actual = {path.relative_to(support).as_posix(): path for path in support.rglob("*")
              if path.is_file() and path != manifest_path}
    errors = []
    for name in sorted(set(entries) - set(actual)):
        errors.append(name + ": listed in manifest but missing from expanded support")
    for name in sorted(set(actual) - set(entries)):
        errors.append(name + ": present in expanded support but missing from manifest")
    for name in sorted(set(entries) & set(actual)):
        item, path = entries[name], actual[name]
        if path.stat().st_size != item.get("bytes") or digest(path) != item.get("sha256"):
            errors.append(name + ": expanded support differs from manifest")
    return errors


def inspect_delivery_freshness(paper_pdf, packaged_pdf, ai_pdf, support, zip_path):
    """Cross-check the working sources, rendered PDFs, support tree, and ZIP."""
    paper_pdf, packaged_pdf, ai_pdf = map(Path, (paper_pdf, packaged_pdf, ai_pdf))
    support, zip_path = Path(support), Path(zip_path)
    pairs = [("paper PDF", paper_pdf, packaged_pdf),
             ("AI details PDF", ai_pdf, support / "AI工具使用详情.pdf")]
    pairs.extend((path.relative_to(ROOT).as_posix(), path,
                  support / path.relative_to(ROOT)) for path in source_files())
    errors = compare_file_pairs(pairs)
    errors.extend(audit_support(support)["errors"])
    if paper_pdf.is_file() and (ROOT / "paper/main.tex").stat().st_mtime > paper_pdf.stat().st_mtime + 1e-6:
        errors.append("current paper PDF predates its source")
    if ai_pdf.is_file() and (ROOT / "paper/ai_details.tex").stat().st_mtime > ai_pdf.stat().st_mtime + 1e-6:
        errors.append("current AI details PDF predates its source")
    if zip_path.is_file():
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for name in expected_support_paths():
                    if archive.read(name) != (support / name).read_bytes():
                        errors.append(name + ": ZIP differs from expanded support")
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            errors.append("ZIP freshness check failed: " + str(exc))
    return {"passed": not errors, "errors": errors}


def make_zip(support, target):
    temporary = target.with_name("." + target.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=False) as archive:
        for path in sorted(support.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.relative_to(support).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    size = temporary.stat().st_size
    if size >= LIMIT_BYTES:
        temporary.unlink()
        raise RuntimeError("support ZIP requires {} bytes, exceeding the strict limit; no files were silently dropped".format(size))
    check = inspect_zip(temporary)
    if not check["passed"]:
        raise RuntimeError("ZIP validation failed: " + "; ".join(check["errors"]))
    temporary.replace(target)
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="list planned source copies without writing or compiling")
    modes.add_argument("--check-only", action="store_true", help="read-only validation of existing deliverables")
    modes.add_argument("--no-compile", action="store_true", help="package already-current paper PDFs without recompiling")
    args = parser.parse_args()
    output = ROOT / "final-output"
    generated_results = ROOT / "artifacts/exports"
    paper_pdf, zip_path = output / "paper.pdf", output / "supporting-materials.zip"
    if args.dry_run:
        print(json.dumps({"source_files": [str(path.relative_to(ROOT)) for path in source_files()], "existing_result_bytes": sum(path.stat().st_size for path in generated_results.rglob("*") if path.is_file()), "strict_limit_bytes_per_deliverable": LIMIT_BYTES, "mode": "read_only_dry_run"}, ensure_ascii=False, indent=2))
        return
    if args.check_only:
        with tempfile.TemporaryDirectory(prefix=".delivery-check-", dir=output) as temporary:
            support = Path(temporary) / "support"
            support.mkdir()
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(support)
            report = {"paper": inspect_pdf(paper_pdf),
                      "current_paper": inspect_pdf(ROOT / "paper/main.pdf"),
                      "ai_details": inspect_pdf(support / "AI工具使用详情.pdf", "ai"),
                      "support": audit_support(support), "zip": inspect_zip(zip_path),
                      "freshness": inspect_delivery_freshness(ROOT / "paper/main.pdf", paper_pdf,
                                                               ROOT / "paper/ai_details.pdf", support, zip_path)}
            passed = all(report[key]["passed"] for key in
                         ("paper", "current_paper", "ai_details", "support", "zip", "freshness"))
            print(json.dumps({"passed": passed, **report}, ensure_ascii=False, indent=2))
            if not passed:
                raise RuntimeError("existing delivery checks failed")
        return
    verification = json.loads((ROOT / "artifacts/verification.json").read_text(encoding="utf-8"))
    if not verification.get("passed") or verification.get("workbooks_checked") != 5:
        raise RuntimeError("run full verification with all five workbooks before packaging")
    verify_experiment_inputs()
    test_check = run_tests()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".support-staging-", dir=output) as temporary:
        staging = Path(temporary)
        copy_support(staging)
        compile_reports = [] if args.no_compile else [compile_tex("ai_details"), compile_tex("main")]
        paper_check = inspect_pdf(ROOT / "paper/main.pdf")
        ai_check = inspect_pdf(ROOT / "paper/ai_details.pdf", "ai")
        report = {"paper": paper_check, "ai_details": ai_check, "tests": test_check, "compilation": compile_reports, "strict_limit_bytes_per_deliverable": LIMIT_BYTES, "target_bytes_per_deliverable": TARGET_BYTES, "visual_review": "Automated checks do not replace inspection of rendered pages."}
        if not paper_check["passed"] or not ai_check["passed"]:
            compact_json(ROOT / "artifacts/package_report.json", report)
            raise RuntimeError("PDF structure checks failed: " + "; ".join(paper_check["errors"] + ai_check["errors"]))
        shutil.copy2(ROOT / "paper/ai_details.pdf", staging / "AI工具使用详情.pdf")
        support_check = audit_support(staging)
        report["support"] = support_check
        if not support_check["passed"]:
            compact_json(ROOT / "artifacts/package_report.json", report)
            raise RuntimeError("support checks failed: " + "; ".join(support_check["errors"]))
        try:
            zip_report = make_zip(staging, zip_path)
        except (OSError, ValueError, RuntimeError) as exc:
            report.update(passed=False, error=str(exc).replace(str(ROOT), "."))
            compact_json(ROOT / "artifacts/package_report.json", report)
            raise
        shutil.copy2(ROOT / "paper/main.pdf", paper_pdf)
        freshness = inspect_delivery_freshness(ROOT / "paper/main.pdf", paper_pdf,
                                               ROOT / "paper/ai_details.pdf", staging, zip_path)
    if not freshness["passed"]:
        raise RuntimeError("freshness checks failed after packaging: " + "; ".join(freshness["errors"]))
    report.update(passed=True, support_files=support_check["files"], zip=zip_report,
                  freshness=freshness, outputs=["final-output/paper.pdf", "final-output/supporting-materials.zip"])
    compact_json(ROOT / "artifacts/package_report.json", report)
    for path in (paper_pdf, zip_path):
        print("{} | {} bytes | SHA256 {}".format(path.relative_to(ROOT), path.stat().st_size, digest(path)), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile, subprocess.TimeoutExpired) as error:
        print("Packaging failed: " + str(error).replace(str(ROOT), "."), file=sys.stderr)
        sys.exit(1)
