"""Clear executable-stack flags from vLLM's compiled extensions.

Background
----------
Since glibc 2.41, ``dlopen`` refuses to load shared objects whose
``PT_GNU_STACK`` program header requests an executable stack. The vLLM CPU
wheels (observed with 0.27.1) ship ``vllm/_C.abi3.so`` with ``GNU_STACK =
RWE``, so ``vllm.LLM(...)`` fails with::

    cannot enable executable stack as shared object requires: Invalid argument

The engine then starts without its ``_C`` extension and the CPU worker exits,
which surfaces as ``Engine core initialization failed`` in the tests.

This script walks the installed vLLM package, inspects every ELF ``*.so*``
file, and clears the executable-stack flag with
``patchelf --clear-execstack``. Clearing the flag is safe when the library does
not actually execute code on the stack, which is normally the case: the flag
is a build leftover (an assembly file missing ``.note.GNU-stack``).

Usage
-----
    python .github/workflows/patch_vllm_execstack.py <vllm-package-dir>

``patchelf`` must be on ``PATH``, for example from
``sudo apt-get install --no-install-recommends patchelf``. The patch must be
reapplied after vLLM is reinstalled. See ``.github/workflows/README.md``.
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess
import sys

PT_GNU_STACK = 0x6474E551
PF_X = 0x1
ELF_MAGIC = b"\x7fELF"
ELFCLASS64 = 2


def has_executable_stack(path: pathlib.Path) -> bool:
    """Check whether a shared object requests an executable stack.

    Parameters
    ----------
    path:
        Candidate shared object file.

    Returns
    -------
    bool
        ``True`` if the file is a 64-bit ELF object with a ``PT_GNU_STACK``
        program header whose executable bit is set. Non-ELF and 32-bit files
        return ``False``.

    Notes
    -----
    This reads the ELF program headers directly so the check does not depend
    on ``readelf`` being installed.
    """
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) < 64:
            return False
        if header[:4] != ELF_MAGIC or header[4] != ELFCLASS64:
            return False

        phoff = struct.unpack_from("<Q", header, 0x20)[0]
        phentsize, phnum = struct.unpack_from("<HH", header, 0x36)

        for index in range(phnum):
            stream.seek(phoff + index * phentsize)
            program_header = stream.read(8)
            if len(program_header) < 8:
                return False
            p_type, p_flags = struct.unpack("<II", program_header)
            if p_type == PT_GNU_STACK:
                return bool(p_flags & PF_X)

    return False


def main(argv: list[str] | None = None) -> int:
    """Patch all vLLM extensions that request an executable stack.

    Parameters
    ----------
    argv:
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code. ``0`` means every found extension has a
        non-executable stack; ``1`` means at least one flag remains or
        ``patchelf`` failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package_dir",
        type=pathlib.Path,
        help="Path to the installed vllm package directory.",
    )
    parser.add_argument(
        "--patchelf",
        default="patchelf",
        help="Path to the patchelf executable (default: found on PATH).",
    )
    args = parser.parse_args(argv)

    if not args.package_dir.is_dir():
        parser.error(f"not a directory: {args.package_dir}")

    extensions = sorted(
        path for path in args.package_dir.rglob("*.so*") if path.is_file()
    )
    if not extensions:
        print(f"No shared objects found under {args.package_dir}; nothing to patch.")
        return 0

    patched = 0
    for extension in extensions:
        if not has_executable_stack(extension):
            continue
        print(f"clearing executable-stack flag: {extension}")
        try:
            subprocess.run(
                [args.patchelf, "--clear-execstack", str(extension)],
                check=True,
            )
        except subprocess.CalledProcessError as error:
            print(f"patchelf failed for {extension}: {error}", file=sys.stderr)
            return 1
        patched += 1

    remaining = [path for path in extensions if has_executable_stack(path)]
    if remaining:
        for path in remaining:
            print(f"executable-stack flag still set: {path}", file=sys.stderr)
        return 1

    print(
        f"Checked {len(extensions)} vLLM extension(s); "
        f"cleared {patched} executable-stack flag(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
