#!/usr/bin/env python3
"""Build the native bridge against a fingerprinted, previously validated runtime.

This reuses version-pinned libraries only when the entire src tree is identical.
A changed core requires a fresh validated core build, not this incremental bridge.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-run", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    previous, source, output = (p.resolve() for p in (args.validated_run, args.source, args.output))
    run = json.loads((previous / "run.json").read_text())
    tested = run["committed_as"]
    def git(*items):
        return subprocess.check_output(["git", "-C", str(source), *items], text=True).strip()
    if git("status", "--porcelain", "--", "src") or git("rev-parse", "HEAD:src") != git("rev-parse", tested + ":src"):
        raise SystemExit("Source differs from the validated runtime; build and validate core first")
    if Path(run["python"]).resolve() != Path(sys.executable).resolve():
        raise SystemExit("Use the Python interpreter recorded by the validated run")
    fingerprints = json.loads((previous / "runtime-fingerprints.json").read_text())
    for path, expected in fingerprints.items():
        if digest(path) != expected["sha256"]:
            raise SystemExit("Validated runtime fingerprint changed: " + path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "lib").mkdir()
    prior = (previous / "candidate").resolve()
    for library in (prior / "lib").iterdir():
        (output / "lib" / library.name).symlink_to(library.resolve())
    cpp = Path(__file__).resolve().parents[1] / "ann_benchmarks/algorithms/zvec_qg/core.cpp"
    compile_entry = next(x for x in json.loads((prior / "build-plan.json").read_text())["compiles"] if x["name"] == "direct")
    link_entry = json.loads((prior / "link-plan.json").read_text())[-1]
    commands = []
    for entry, kind in [(compile_entry, "compile"), (link_entry, "link")]:
        command = []
        for item in entry["args"]:
            if item.startswith("-Wl,--dependency-file=") or item.startswith("-Wl,-rpath,"):
                continue
            item = item.replace(str(prior), str(output)).replace(str(previous / "direct.cpp"), str(cpp))
            item = item.replace(str(run["source"]) + "/src", str(source) + "/src")
            command.append(item)
        if kind == "link":
            command.append("-Wl,-rpath," + str(output / "lib"))
        commands.append(dict(args=command, cwd=entry["cwd"], kind=kind))
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        with (output / (kind + ".log")).open("w") as log:
            subprocess.run(command, cwd=entry["cwd"], stdout=log, stderr=subprocess.STDOUT, check=True)
    extension = output / ("_direct_core" + sysconfig.get_config_var("EXT_SUFFIX"))
    runtime = [extension] + [p for p in (output / "lib").iterdir() if p.is_file()]
    record = dict(source_commit=git("rev-parse", "HEAD"), source_tree=git("rev-parse", "HEAD:src"),
                  source_branch=git("branch", "--show-current"), validated_core_commit=tested,
                  validated_run=str(previous), extension=str(extension), python=sys.executable,
                  bridge_sha256=digest(cpp), runtime_sha256={str(p): digest(p) for p in runtime})
    (output / "artifact.json").write_text(json.dumps(record, indent=2) + "\n")
    print(output / "artifact.json")


if __name__ == "__main__":
    main()
