import os
import shutil


WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_PRECURSOR_BASE = os.path.join(WORKSPACE_ROOT, "Solver", "ADM", "precursor_Base")


def _coerce_precursor_root(cfg):
    env_root = os.environ.get("WINDRL_PRECURSOR_ROOT")
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))

    if cfg is not None:
        root = getattr(cfg.env, "precursor_root", None)
        if root:
            return os.path.abspath(os.path.expanduser(str(root)))

    return None


def ensure_precursor(instance, cfg=None):
    workspace_precursor_dir = os.path.join(WORKSPACE_ROOT, "Solver", "ADM", f"precursor_{instance}")
    source_root = _coerce_precursor_root(cfg)

    if source_root:
        source_dir = os.path.join(source_root, f"precursor_{instance}")
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(
                f"Missing precursor directory for instance {instance}: {source_dir}"
            )

        if os.path.lexists(workspace_precursor_dir):
            if os.path.islink(workspace_precursor_dir) or os.path.isfile(workspace_precursor_dir):
                os.unlink(workspace_precursor_dir)
            else:
                shutil.rmtree(workspace_precursor_dir)

        os.symlink(source_dir, workspace_precursor_dir)
        return workspace_precursor_dir

    if not os.path.exists(workspace_precursor_dir):
        if os.path.isdir(LOCAL_PRECURSOR_BASE):
            shutil.copytree(LOCAL_PRECURSOR_BASE, workspace_precursor_dir, symlinks=True)
        else:
            raise FileNotFoundError(f"Missing precursor base directory: {LOCAL_PRECURSOR_BASE}")

    return workspace_precursor_dir
