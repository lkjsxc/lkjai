#!/usr/bin/env python3
import re
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[2]
    cmake = read(repo / "native" / "CMakeLists.txt")
    require("LKJAI_DEFAULT_CUDA_ARCHITECTURES" in cmake, "missing default list")
    for arch in ["86-real", "86-virtual", "89-real", "89-virtual",
                 "90-real", "90-virtual", "120-real", "120-virtual"]:
        require(arch in cmake, f"default arch missing {arch}")
    require("DEFINED CMAKE_CUDA_ARCHITECTURES" in cmake, "missing cmake precedence")
    require("ENV{LKJAI_CUDA_ARCHS}" in cmake, "missing LKJAI_CUDA_ARCHS support")
    require(
        re.search(r"CMAKE_CUDA_ARCHITECTURES.*LKJAI_DEFAULT_CUDA_ARCHITECTURES", cmake),
        "default arch list is not assigned to CMAKE_CUDA_ARCHITECTURES",
    )
    for dockerfile in [
        repo / "ops" / "docker" / "Dockerfile.native",
        repo / "ops" / "docker" / "Dockerfile.verify",
    ]:
        text = read(dockerfile)
        require("ARG LKJAI_CUDA_ARCHS" in text, f"{dockerfile}: missing ARG")
        require("ENV LKJAI_CUDA_ARCHS=" in text, f"{dockerfile}: missing ENV")
    compose = read(repo / "compose.yaml")
    require("LKJAI_CUDA_ARCHS: ${LKJAI_CUDA_ARCHS:-}" in compose, "compose missing build arg")
    require("LKJAI_CUDA_ARCHS: ${LKJAI_CUDA_ARCHS:-}" in compose, "compose missing env")


if __name__ == "__main__":
    main()
