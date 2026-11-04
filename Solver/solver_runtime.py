import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORK = "github.com/rklopsch/Incompact3d-WindRL"


@lru_cache(maxsize=4)
def _verified_build(root, build):
    root = Path(root)
    build = Path(build)
    remote = subprocess.check_output(
        ["git", "-C", str(root), "remote", "get-url", "origin"], text=True
    ).strip().removesuffix(".git").replace("git@github.com:", "github.com/").removeprefix("https://")
    if remote != FORK:
        raise ValueError(f"Solver origin must be {FORK}; got {remote}")
    cache = build / "CMakeCache.txt"
    if not cache.is_file():
        raise FileNotFoundError(f"Configure and build the fork first: {build}")
    settings = {}
    for line in cache.read_text().splitlines():
        if "=" in line and ":" in line.split("=", 1)[0]:
            key, value = line.split("=", 1)
            settings[key.split(":", 1)[0]] = value
    if Path(settings.get("CMAKE_HOME_DIRECTORY", "")).resolve() != root:
        raise ValueError(f"{cache} was configured for another source checkout")
    executable = (build / "bin" / "xcompact3d").resolve()
    if not executable.is_relative_to(build) or not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"Missing fork executable: {executable}")
    verifier = root / "tools" / "solver_manifest.py"
    if not verifier.is_file():
        raise ValueError("The fork is missing the CSV/pressure implementation; check out the matching solver revision")
    result = subprocess.run(
        [sys.executable, str(verifier), "--binary", str(executable), "--verify"],
        text=True, capture_output=True,
    )
    if result.returncode:
        raise ValueError(f"Fork solver verification failed; rebuild it. {result.stderr.strip()}")
    metadata = json.loads(result.stdout)
    metadata.update(executable=str(executable), source=str(root), origin=remote)
    library = settings.get("SMARTREDIS_LIBRARY", "")
    return str(executable), metadata, str(Path(library).parent) if library else ""


def verified_solver(env):
    root = Path(os.environ.get("WINDRL_SOLVER_ROOT", env.get("solver_root", "../Incompact3d-WindRL"))).expanduser()
    root = (PROJECT_ROOT / root).resolve()
    build = (root / str(env.get("solver_build", "build"))).resolve()
    executable, metadata, library = _verified_build(str(root), str(build))
    if library:
        paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if library not in paths:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([library, *filter(None, paths)])
    return executable, metadata


def record_solver(directory, env):
    _, metadata = verified_solver(env)
    (Path(directory) / "solver_build.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")