from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from ibm650_it import REPO_ROOT


def load_runpod_env(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    env = os.environ.copy()
    env_path = repo_root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key.strip(), value.strip())
    if "RUNPOD_API_KEY" not in env:
        raise RuntimeError("RUNPOD_API_KEY is not set; expected it in environment or .env")
    return env


class RunpodCtl:
    def __init__(
        self,
        *,
        repo_root: Path = REPO_ROOT,
        ssh_key_path: Path | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.env = load_runpod_env(repo_root)
        self.ssh_key_path = ssh_key_path or Path.home() / ".ssh" / "id_ed25519"

    def _run(self, args: list[str], *, json_output: bool = True, check: bool = True) -> Any:
        command = ["runpodctl"]
        if json_output:
            command.extend(["-o", "json"])
        command.extend(args)
        proc = subprocess.run(
            command,
            cwd=self.repo_root,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout)
        if not json_output:
            return proc.stdout
        return json.loads(proc.stdout)

    def ensure_ssh_key(self, key_file: Path | None = None) -> dict[str, Any]:
        key_file = key_file or self.ssh_key_path.with_suffix(".pub")
        return self._run(["ssh", "add-key", "--key-file", str(key_file)])

    def list_gpus(self) -> list[dict[str, Any]]:
        return self._run(["gpu", "list"])

    def list_pods(self, *, all_pods: bool = False) -> list[dict[str, Any]]:
        args = ["pod", "list"]
        if all_pods:
            args.append("--all")
        return self._run(args)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self._run(["pod", "get", pod_id])

    def create_pod(
        self,
        *,
        name: str,
        gpu_id: str,
        image: str = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        cloud_type: str = "COMMUNITY",
        ports: str = "22/tcp,8888/http",
        container_disk_gb: int = 100,
        volume_gb: int = 100,
        volume_mount_path: str = "/workspace",
        public_ip: bool = True,
    ) -> dict[str, Any]:
        args = [
            "pod",
            "create",
            "--cloud-type",
            cloud_type,
            "--gpu-id",
            gpu_id,
            "--name",
            name,
            "--image",
            image,
            "--ports",
            ports,
            "--container-disk-in-gb",
            str(container_disk_gb),
            "--volume-in-gb",
            str(volume_gb),
            "--volume-mount-path",
            volume_mount_path,
        ]
        if public_ip and cloud_type.upper() == "COMMUNITY":
            args.append("--public-ip")
        args.append("--ssh")
        return self._run(args)

    def delete_pod(self, pod_id: str) -> dict[str, Any]:
        return self._run(["pod", "delete", pod_id])

    def ssh_info(self, pod_id: str) -> dict[str, Any]:
        return self._run(["ssh", "info", pod_id])

    def wait_for_ssh(self, pod_id: str, *, timeout_seconds: int = 600, poll_seconds: int = 10) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            info = self.ssh_info(pod_id)
            if "error" not in info:
                return info
            time.sleep(poll_seconds)
        raise TimeoutError(f"pod {pod_id} did not become ssh-ready within {timeout_seconds}s")

    @staticmethod
    def _ssh_target(info: dict[str, Any]) -> tuple[str, str, str]:
        if "ip" in info and "port" in info:
            return "root", str(info["ip"]), str(info["port"])
        command = info.get("ssh_command")
        if not command:
            raise KeyError(f"ssh info missing ip/port and ssh_command: {info}")
        tokens = shlex.split(str(command))
        user_host = next(token for token in tokens if "@" in token and not token.startswith("-"))
        user, host = user_host.split("@", 1)
        port = "22"
        for index, token in enumerate(tokens):
            if token == "-p" and index + 1 < len(tokens):
                port = tokens[index + 1]
                break
        return user, host, port

    def ssh(
        self,
        info: dict[str, Any],
        remote_command: str,
        *,
        check: bool = True,
        yield_time_ms: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        user, host, port = self._ssh_target(info)
        return subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-i",
                str(self.ssh_key_path),
                "-p",
                port,
                f"{user}@{host}",
                remote_command,
            ],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=check,
        )

    def scp_to(self, info: dict[str, Any], local_path: Path, remote_path: str) -> subprocess.CompletedProcess[str]:
        user, host, port = self._ssh_target(info)
        return subprocess.run(
            [
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-i",
                str(self.ssh_key_path),
                "-P",
                port,
                "-r",
                str(local_path),
                f"{user}@{host}:{remote_path}",
            ],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=True,
        )

    def scp_from(self, info: dict[str, Any], remote_path: str, local_path: Path) -> subprocess.CompletedProcess[str]:
        user, host, port = self._ssh_target(info)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        return subprocess.run(
            [
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-i",
                str(self.ssh_key_path),
                "-P",
                port,
                "-r",
                f"{user}@{host}:{remote_path}",
                str(local_path),
            ],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=True,
        )
