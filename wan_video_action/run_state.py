import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone

from accelerate.utils import broadcast_object_list


RUN_MANIFEST_NAME = ".bwm_run.json"
RUN_MANIFEST_SCHEMA_VERSION = 1
RNG_SCHEME = "global_model_init_rank_offset_training_v1"

TRAINING_CODE_PATHS = (
    "scripts/train.py",
    "wan_video_action/data/operators.py",
    "wan_video_action/data/wan_dataset.py",
    "wan_video_action/logger.py",
    "wan_video_action/loss.py",
    "wan_video_action/models/wan_video_action_encoder.py",
    "wan_video_action/parsers.py",
    "wan_video_action/pipelines/wan_video_action.py",
    "wan_video_action/run_state.py",
    "wan_video_action/runner.py",
    "wan_video_action/utils.py",
)


def _sha256_file(path):
    if path is None:
        return None
    path = os.path.abspath(os.fspath(path))
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _path_record(path, include_sha256=False, include_mtime=False):
    if path is None:
        return None
    resolved = os.path.abspath(os.path.expanduser(os.fspath(path)))
    record = {"path": resolved}
    if os.path.isfile(resolved):
        stat = os.stat(resolved)
        record["size"] = stat.st_size
        if include_mtime:
            record["mtime_ns"] = stat.st_mtime_ns
        if include_sha256:
            record["sha256"] = _sha256_file(resolved)
    else:
        record["missing"] = True
    return record


def _stable_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flatten_paths(items):
    flattened = []
    for item in items or []:
        if isinstance(item, (list, tuple)):
            flattened.extend(_flatten_paths(item))
        else:
            flattened.append(item)
    return flattened


def _git_metadata(repo_root):
    metadata = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        metadata.update({"commit": commit, "dirty": bool(status.strip())})
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


def _training_code_identity(repo_root):
    files = {}
    for relative_path in TRAINING_CODE_PATHS:
        path = os.path.join(repo_root, relative_path)
        files[relative_path] = _path_record(path, include_sha256=True)
    return {
        "files": files,
        "combined_sha256": _stable_sha256(files),
    }


def _runtime_identity():
    packages = {}
    for package_name in ("accelerate", "deepspeed", "diffsynth", "torch"):
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = None
    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "packages": packages,
    }


def build_training_identity(args, world_size, repo_root=None):
    """Build the immutable portion of a run configuration used for safe resume."""
    identity = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "rng_scheme": RNG_SCHEME,
        "dataset": {
            "name": args.dataset_name,
            "base_path": os.path.abspath(os.path.expanduser(args.dataset_base_path)),
            "metadata": _path_record(args.dataset_metadata_path, include_sha256=True),
            "repeat": int(args.dataset_repeat),
            "num_workers": int(args.dataset_num_workers),
            "data_keys": list(args.data_keys),
        },
        "model": {
            "model_paths": os.path.abspath(os.path.expanduser(args.model_paths)),
            "model_config": _path_record(args.model_config_path, include_sha256=True),
            "resolved_model_files": [
                _path_record(path, include_mtime=True)
                for path in _flatten_paths(getattr(args, "resolved_model_paths", []))
            ],
            "weights": list(args.weights),
            "modes": dict(args.modes),
            "trainable": list(args.trainable),
            "task": args.task,
            "action_type": args.action_type,
            "action_dim": int(args.action_dim),
            "action_stat": _path_record(args.action_stat_path, include_sha256=True),
        },
        "video": {
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
            "num_history_frames": int(args.num_history_frames),
            "history_template_sampling": int(args.history_template_sampling),
            "time_division_factor": int(args.time_division_factor),
            "time_division_remainder": int(args.time_division_remainder),
            "spatial_division_factor": int(args.spatial_division_factor),
            "resize_mode": args.resize_mode,
        },
        "optimization": {
            "seed": int(args.seed),
            "world_size": int(world_size),
            "batch_size_per_rank": int(args.batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "max_grad_norm": float(args.max_grad_norm),
            "mixed_precision": args.mixed_precision,
            "deterministic": bool(args.deterministic),
            "find_unused_parameters": bool(args.find_unused_parameters),
            "use_gradient_checkpointing": bool(args.use_gradient_checkpointing),
            "use_gradient_checkpointing_offload": bool(args.use_gradient_checkpointing_offload),
            "min_timestep_boundary": float(args.min_timestep_boundary),
            "max_timestep_boundary": float(args.max_timestep_boundary),
        },
        "sampler": {
            "shuffle": True,
            "seedable_sampler": True,
            "data_seed": int(args.seed),
            "even_batches": True,
        },
        "checkpoint": {
            "deepspeed_module_parameters": "trainable_only",
            "completed_state_retention": 2,
        },
        "runtime": _runtime_identity(),
    }
    if repo_root is not None:
        identity["training_code"] = _training_code_identity(repo_root)
    return identity


def _atomic_write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
    directory_fd = os.open(os.path.dirname(path), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _describe_identity_difference(saved, current):
    differences = []

    def visit(prefix, left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                path = f"{prefix}.{key}" if prefix else key
                if key not in left:
                    differences.append(f"{path}: saved=<missing>, current={right[key]!r}")
                elif key not in right:
                    differences.append(f"{path}: saved={left[key]!r}, current=<missing>")
                else:
                    visit(path, left[key], right[key])
            return
        if left != right:
            differences.append(f"{prefix}: saved={left!r}, current={right!r}")

    visit("", saved, current)
    return differences


def _validate_resume_state_path(resume_from):
    resolved = os.path.realpath(os.path.abspath(os.path.expanduser(resume_from)))
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"Resume state directory does not exist: {resume_from}")
    complete_marker = os.path.join(resolved, "COMPLETE")
    if not os.path.isfile(complete_marker):
        raise RuntimeError(
            f"Resume state is missing its COMPLETE marker: {resolved}. "
            "Refusing to load an incomplete or legacy state without --allow_unsafe_resume."
        )
    return resolved


def prepare_run_directory(accelerator, args, repo_root):
    """Create or validate the run manifest, broadcasting one result to every rank."""
    payload = None
    if accelerator.is_main_process:
        try:
            output_path = os.path.abspath(os.path.expanduser(args.output_path))
            identity = build_training_identity(args, accelerator.num_processes, repo_root=repo_root)
            fingerprint = _stable_sha256(identity)
            manifest_path = os.path.join(output_path, RUN_MANIFEST_NAME)
            warnings = []

            if args.resume_from:
                if not args.allow_unsafe_resume:
                    _validate_resume_state_path(args.resume_from)
                elif not os.path.isdir(os.path.realpath(os.path.abspath(args.resume_from))):
                    raise FileNotFoundError(f"Resume state directory does not exist: {args.resume_from}")

                if not os.path.isfile(manifest_path):
                    if not args.allow_unsafe_resume:
                        raise FileNotFoundError(
                            f"Run manifest is missing: {manifest_path}. "
                            "Use --allow_unsafe_resume only for a deliberately audited legacy checkpoint."
                        )
                    manifest = {
                        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                        "run_id": f"unsafe-legacy-{uuid.uuid4()}",
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "config_fingerprint": fingerprint,
                        "training_identity": identity,
                        "legacy_resume": True,
                    }
                    _atomic_write_json(manifest_path, manifest)
                    warnings.append("Resuming without an existing run manifest because --allow_unsafe_resume was set.")
                else:
                    with open(manifest_path, "r", encoding="utf-8") as handle:
                        manifest = json.load(handle)
                    saved_identity = manifest.get("training_identity")
                    saved_fingerprint = manifest.get("config_fingerprint")
                    if saved_identity is None or saved_fingerprint is None:
                        if not args.allow_unsafe_resume:
                            raise RuntimeError(f"Run manifest is missing identity fields: {manifest_path}")
                        warnings.append("Run manifest identity is incomplete; unsafe resume override is active.")
                        manifest["training_identity"] = identity
                        manifest["config_fingerprint"] = fingerprint
                        manifest.setdefault("run_id", f"unsafe-legacy-{uuid.uuid4()}")
                        manifest["unsafe_resume_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                        _atomic_write_json(manifest_path, manifest)
                    elif saved_fingerprint != fingerprint:
                        differences = _describe_identity_difference(saved_identity, identity)
                        detail = "\n".join(f"  - {item}" for item in differences[:20])
                        message = f"Resume configuration does not match the original run:\n{detail}"
                        if len(differences) > 20:
                            message += f"\n  - ... {len(differences) - 20} more difference(s)"
                        if not args.allow_unsafe_resume:
                            raise RuntimeError(message)
                        warnings.append(message)
                        history = manifest.setdefault("unsafe_resume_history", [])
                        history.append(
                            {
                                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                                "previous_config_fingerprint": saved_fingerprint,
                                "reason": "configuration_mismatch_override",
                            }
                        )
                        manifest["training_identity"] = identity
                        manifest["config_fingerprint"] = fingerprint
                        _atomic_write_json(manifest_path, manifest)
            else:
                if os.path.isdir(output_path):
                    existing_entries = sorted(os.listdir(output_path))
                    if existing_entries:
                        preview = ", ".join(existing_entries[:8])
                        raise FileExistsError(
                            f"Fresh training output directory is not empty: {output_path} ({preview}). "
                            "Choose a new OUTPUT_PATH; files are never overwritten automatically."
                        )
                elif os.path.exists(output_path):
                    raise FileExistsError(f"Fresh training output path is not a directory: {output_path}")
                else:
                    os.makedirs(output_path, exist_ok=False)

                manifest = {
                    "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                    "run_id": str(uuid.uuid4()),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config_fingerprint": fingerprint,
                    "training_identity": identity,
                    "initialization_checkpoint": _path_record(args.ckpt_path, include_sha256=False),
                    "git": _git_metadata(repo_root),
                }
                _atomic_write_json(manifest_path, manifest)

            payload = {"ok": True, "manifest": manifest, "warnings": warnings}
        except Exception as exc:
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    objects = [payload]
    broadcast_object_list(objects, from_process=0)
    payload = objects[0]
    if not payload["ok"]:
        raise RuntimeError(payload["error"])
    for warning in payload.get("warnings", []):
        accelerator.print(f"[unsafe_resume_warning] {warning}")
    return payload["manifest"]
