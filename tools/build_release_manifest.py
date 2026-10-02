#!/usr/bin/env python3
"""Generate/check complete release checksums, leaving historical seals untouched."""
import argparse
import csv
import hashlib
import io
from pathlib import Path
import subprocess

EXCLUDED = {
    "provenance/RELEASE_MANIFEST_v1.0.csv",
    "provenance/RELEASE_SHA256_v1.0.txt",
}


def release_files(root):
    # In a checkout, release files are tracked plus nonignored new files.
    # Runtime caches and ignored outputs are not part of the release.
    if (root / ".git").exists():
        data = subprocess.check_output([
            "git", "-C", str(root), "ls-files", "-z", "--cached",
            "--others", "--exclude-standard",
        ])
        names = {x.decode("utf-8") for x in data.split(b"\0") if x}
    else:
        names = {str(p.relative_to(root)).replace("\\", "/")
                 for p in root.rglob("*") if p.is_file() and ".git" not in p.parts}
    return sorted(names - EXCLUDED)


def render_manifests(root):
    csv_stream = io.StringIO(newline="")
    writer = csv.writer(csv_stream, lineterminator="\n")
    writer.writerow(["path", "bytes", "sha256", "git_blob_sha1"])
    checksums = []
    for name in release_files(root):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing or unsupported release file: {name}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        writer.writerow([name, len(data), digest, blob])
        checksums.append(f"{digest}  {name}\n")
    return {
        "provenance/RELEASE_MANIFEST_v1.0.csv": csv_stream.getvalue(),
        "provenance/RELEASE_SHA256_v1.0.txt": "".join(checksums),
    }


def process(root, check=False):
    rendered = render_manifests(root)
    for name, text in rendered.items():
        path = root / name
        if check:
            if not path.exists() or path.read_text() != text:
                raise RuntimeError(f"Release coverage/checksum mismatch: {name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    print("[PASS] complete release manifest" if check else "Release manifest generated")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    process(args.root.resolve(), args.check)


if __name__ == "__main__":
    main()
