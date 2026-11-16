from __future__ import annotations

import os
import socket
import stat
import uuid
from pathlib import Path

import torch


def open_isolated_root(repo_root: Path, rel_dir: str | Path) -> int:
    base = Path(repo_root).resolve()
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        for part in Path(rel_dir).parts:
            if part in ("..", "/"):
                raise RuntimeError(
                    f"open_isolated_root: refusing component {part!r} in {rel_dir}")
            try:
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY
                              | getattr(os, "O_NOFOLLOW", 0), dir_fd=fd)
            except FileNotFoundError:
                os.mkdir(part, 0o755, dir_fd=fd)
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY
                              | getattr(os, "O_NOFOLLOW", 0), dir_fd=fd)
            except OSError as e:
                import errno as _errno
                if e.errno in (_errno.ELOOP, _errno.ENOTDIR):
                    raise RuntimeError(
                        f"open_isolated_root: {part!r} under {rel_dir} is a "
                        "symlink or non-directory — refusing (write-redirect "
                        "hazard)") from None
                raise
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def publish_no_replace_at(dirfd: int, name: str, text: str) -> None:
    if "/" in name or name in ("", ".", ".."):
        raise RuntimeError(f"publish_no_replace_at: bad final name {name!r}")
    tmp = f"{name}.tmp.{uuid.uuid4().hex}"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(tmp, flags, 0o644, dir_fd=dirfd)
    try:
        _write_all(fd, text.encode())
        os.fsync(fd)
        try:
            os.link(tmp, name, src_dir_fd=dirfd, dst_dir_fd=dirfd,
                    follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(
                f"{name} already exists in the registered directory — "
                "refusing to overwrite a published result"
            ) from None
    finally:
        os.close(fd)
        try:
            os.unlink(tmp, dir_fd=dirfd)
        except FileNotFoundError:
            pass


def enforce_single_visible_gpu(device: torch.device) -> None:
    if device.type != "cuda":
        return
    n = torch.cuda.device_count()
    if n != 1:
        raise RuntimeError(
            f"{n} CUDA devices visible; the checkpoint-restored RNG protocol "
            "requires exactly one (set CUDA_VISIBLE_DEVICES=<index> and relaunch)"
        )


def publish_no_replace_isolated(repo_root: Path, path: Path, text: str) -> None:
    rp = Path(path).resolve()
    root = Path(repo_root).resolve()
    try:
        rel = rp.relative_to(root)
    except ValueError:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        publish_no_replace(Path(path), text)
        return
    dirfd = open_isolated_root(root, rel.parent)
    try:
        publish_no_replace_at(dirfd, rel.name, text)
    finally:
        os.close(dirfd)


def claim_output_lock_isolated(repo_root: Path, out_path: Path) -> tuple[Path, str]:
    rp = Path(out_path).resolve()
    root = Path(repo_root).resolve()
    try:
        rel = rp.relative_to(root)
    except ValueError:
        return claim_output_lock(Path(out_path))
    dirfd = open_isolated_root(root, rel.parent)
    try:
        lock_name = rel.name + ".lock"
        token = f"{uuid.uuid4()}|{socket.gethostname()}|pid={os.getpid()}"
        try:
            os.mkdir(lock_name, dir_fd=dirfd)
        except FileExistsError:
            raise FileExistsError(
                f"{rp}.lock exists — another job owns this output (or crashed "
                "and left a stale lock; remove it explicitly after checking)"
            ) from None
        ofd = os.open(f"{lock_name}/owner",
                      os.O_CREAT | os.O_EXCL | os.O_WRONLY
                      | getattr(os, "O_NOFOLLOW", 0), 0o644, dir_fd=dirfd)
        try:
            _write_all(ofd, (token + "\n").encode())
        finally:
            os.close(ofd)
        return Path(str(out_path) + ".lock"), token
    finally:
        os.close(dirfd)


def claim_output_lock(out_path: Path) -> tuple[Path, str]:
    lock = Path(str(out_path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    token = f"{uuid.uuid4()}|{socket.gethostname()}|pid={os.getpid()}"
    try:
        lock.mkdir()
    except FileExistsError:
        raise FileExistsError(
            f"{lock} exists — another job owns this output (or crashed and "
            "left a stale lock; remove it explicitly after checking)"
        ) from None
    (lock / "owner").write_text(token + "\n")
    return lock, token


def _lock_owner(lock: Path) -> str | None:
    try:
        return (lock / "owner").read_text().strip()
    except OSError:
        return None


def assert_lock_owned(lock: Path, token: str) -> None:
    current = _lock_owner(lock)
    if current != token:
        raise RuntimeError(
            f"lock ownership lost: {lock} now holds {current!r}, expected our "
            f"token — another job replaced the claim; refusing to publish"
        )


def release_output_lock(lock: Path, token: str) -> None:
    current = _lock_owner(lock)
    if current is None:
        if lock.exists():
            raise RuntimeError(
                f"{lock} exists without an owner sentinel (crashed mid-claim?) "
                "— inspect and remove explicitly"
            )
        return
    if current != token:
        raise RuntimeError(
            f"refusing to release {lock}: it now holds {current!r}, not our token"
        )
    (lock / "owner").unlink()
    lock.rmdir()


def _exclusive_tmp_fd(path: Path) -> tuple[Path, int]:
    tmp = Path(f"{path}.tmp.{uuid.uuid4().hex}")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return tmp, os.open(tmp, flags, 0o644)


def _verify_tmp_is_ours(tmp: Path, fd: int) -> None:
    st_name = os.lstat(tmp)
    st_fd = os.fstat(fd)
    if (st_name.st_ino != st_fd.st_ino or st_name.st_dev != st_fd.st_dev
            or not stat.S_ISREG(st_name.st_mode) or st_fd.st_nlink != 1):
        raise RuntimeError(
            f"{tmp} was tampered with during publication (inode swap) — aborting"
        )


def _safe_unlink_tmp(tmp: Path, ino: int) -> None:
    try:
        st = os.lstat(tmp)
    except FileNotFoundError:
        return
    if st.st_ino == ino:
        tmp.unlink(missing_ok=True)


def _write_all(fd: int, data: bytes) -> None:
    import errno

    view = memoryview(data)
    while view:
        try:
            n = os.write(fd, view)
        except InterruptedError:
            continue
        if n <= 0:
            raise OSError(errno.EIO, f"zero-progress write ({len(view)} bytes left)")
        view = view[n:]


def publish_no_replace(path: Path, text: str) -> None:
    path = Path(path)
    tmp, fd = _exclusive_tmp_fd(path)
    try:
        _write_all(fd, text.encode())
        os.fsync(fd)
        _verify_tmp_is_ours(tmp, fd)
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise FileExistsError(
                f"{path} already exists — refusing to overwrite a published result"
            ) from None
    finally:
        ino = os.fstat(fd).st_ino
        os.close(fd)
        _safe_unlink_tmp(tmp, ino)
