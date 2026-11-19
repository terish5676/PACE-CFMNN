"""Download the UCI and OpenML datasets into data/ and check them against the files the results used.

    python -B scripts/download_data.py                 # download what is missing, then check
    python -B scripts/download_data.py --check         # check only; no network
    python -B scripts/download_data.py --datasets magic letter

Each UCI dataset's zip comes from https://archive.ics.uci.edu/static/public/<id>/<name>.zip. Only
the files the loaders in pace_cfmnn/data.py read are kept. Two of them ship compressed with Unix
`compress` (shuttle.trn.Z, page-blocks.data.Z); they are unpacked with `gzip -dc`, which must be on
the PATH (it is on Linux, macOS and Git Bash for Windows; otherwise unpack them with 7-Zip).

The OpenML datasets (Runs P9 and P10; the rule that chose them is results/heldout/RULE.md) are
single ARFF files from https://openml.org/data/v1/download/<file id>/<name>. Each download is first
checked against the MD5 that OpenML publishes for it; a file that fails is not saved.

Every file is then compared with the SHA-256 of the copy our runs used. A mismatch means the source
changed the file, and the results may not reproduce exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://archive.ics.uci.edu/static/public/{id}/{name}.zip"

# folder: (UCI id, zip name, {file the loader reads: SHA-256 of the copy our runs used})
DATASETS = {
    "thyroid": (102, "thyroid+disease", {
        "ann-train.data": "3da53a156bda36cb0c97e9f4b6b111c9226c54c4aa00230de5604b787c47e3a6",
        "ann-test.data": "c649ea19416e78c7996cfaaa2a9e281cb597d4b075aaa68c494fc3e4ee3aa30b",
    }),
    "shuttle": (148, "statlog+shuttle", {
        "shuttle.trn": "87b24ee9fb5137e1d417659cf905d84d0e15342bbaa60770f1ae83da1a38200a",
        "shuttle.tst": "f776934a628d9b94c482cb058a76ddaa63823e0f74aec065813c9dff3b89d661",
    }),
    "skin": (229, "skin+segmentation", {
        "Skin_NonSkin.txt": "e30c0a845385dcc95a45c45ed263465674a49638e98ef740afd520769c7714a4",
    }),
    "sepsis": (827, "sepsis+survival+minimal+clinical+records", {
        "s41598-020-73558-3_sepsis_survival_primary_cohort.csv":
            "6ba4ebd256fa3b5f6c0d42444a3e6b9a4b90d6e493c9efdfedcb2ef7470d4b0f",
    }),
    "magic": (159, "magic+gamma+telescope", {
        "magic04.data": "e9314b7ebd4b4b59a3b3d65f7316663963777b16a46786877651dbbaa640b36a",
    }),
    "letter": (59, "letter+recognition", {
        "letter-recognition.data": "2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2",
    }),
    "landsat": (146, "statlog+landsat+satellite", {
        "sat.trn": "e896dc88a960fa2404160fc4c3cb3dc53fcf4afd80ba920bf2d261bd42d12613",
        "sat.tst": "4b9167b8a92baafafed7c8809aef86d0683a5e685c19d98d119fa6974f0f2479",
    }),
    "page_blocks": (78, "page+blocks+classification", {
        "page-blocks.data": "52058b6e19ba797b5e80ae6454ebb3b7a25016787e668dd3203393c236d22c78",
    }),
    "covertype": (31, "covertype", {
        "covtype.data.gz": "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771",
    }),
}

OPENML_URL = "https://openml.org/data/v1/download/{file_id}/{name}"

# folder: (OpenML data id, file id, file name, MD5 OpenML publishes, SHA-256 of the copy our runs
# used). Metadata read from https://www.openml.org/api/v1/json/data/<data id> on 2026-11-16.
OPENML = {
    "phoneme": (1489, 1592281, "phoneme.arff", "902ff27649dbfb512f3fb01fc05c6b24",
                "8dcd1c5236cea66cdc4e75cb84fbb4da7a4b1e8b1fe5ad5cd3fd825a1ae4c48b"),
    "wilt": (40983, 18151926, "wilt.arff", "aa7c0aaf3fd671f06bbf6a68be4754f7",
             "e29d2b9c08fbd21fe7104df37f16eed8fd0d3a9c782cbb8efada0cf7ebb8d6bb"),
    "mfeat_morphological": (18, 18, "mfeat-morphological.arff", "5600bd857927b893e2e4430240b2e74a",
                            "0ad5a3afe6ad0a4851d2888980ce099b69c76030801bafab1e0e7e2e284837ea"),
    "jungle_chess": (41027, 18631418, "jungle_chess_2pcs_raw_endgame_complete.arff",
                     "25ca110c24c9dc3ffdcb5c2453382778", "647f9cded1b34558bd4587106bfabcb908e706194e78d3014fea6d594e909d75"),
    "electricity": (151, 2419, "electricity.arff", "8ca97867d960ae029ae3a9ac2c923d34",
                    "2d86fbc74c69a5c04bcde0119bce280dd6d8efa0115987edc75b77ccf024d7f2"),
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def status(path: str, want: str) -> str:
    """'ok', 'missing' or 'changed'."""
    if not os.path.isfile(path):
        return "missing"
    return "ok" if sha256(path) == want else "changed"


def members(zf: zipfile.ZipFile) -> dict[str, bytes]:
    """Every file in a zip by its base name, looking inside zips nested in it."""
    out = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        data = zf.read(info)
        base = os.path.basename(info.filename)
        if base.lower().endswith(".zip"):
            out.update(members(zipfile.ZipFile(io.BytesIO(data))))
        else:
            out[base] = data
    return out


def uncompress(data: bytes, name: str) -> bytes:
    """Unpack a Unix `compress` (.Z) file with gzip, which reads that format."""
    gzip = shutil.which("gzip")
    if gzip is None:
        sys.exit(f"{name} is compressed with Unix compress and gzip is not on the PATH; "
                 "unpack it with 7-Zip and run this script with --check")
    return subprocess.run([gzip, "-dc"], input=data, capture_output=True, check=True).stdout


def fetch(folder: str, uci_id: int, zip_name: str, files: dict[str, str], data_dir: str) -> None:
    url = URL.format(id=uci_id, name=zip_name)
    print(f"{folder}: downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "pace-cfmnn-data-download"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        found = members(zipfile.ZipFile(io.BytesIO(resp.read())))
    os.makedirs(os.path.join(data_dir, folder), exist_ok=True)
    for name in files:
        if name in found:
            data = found[name]
        elif name + ".Z" in found:
            data = uncompress(found[name + ".Z"], name + ".Z")
        else:
            sys.exit(f"{folder}: {name} is not in {url}; the zip's layout has changed")
        with open(os.path.join(data_dir, folder, name), "wb") as f:
            f.write(data)


def fetch_openml(folder: str, file_id: int, name: str, md5: str, data_dir: str) -> None:
    url = OPENML_URL.format(file_id=file_id, name=name)
    print(f"{folder}: downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "pace-cfmnn-data-download"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    got = hashlib.md5(data).hexdigest()
    if got != md5:
        sys.exit(f"{folder}: {name} fails the MD5 OpenML publishes ({got}, expected {md5}); "
                 "not saved. Under results/heldout/RULE.md such a dataset is left out and reported")
    os.makedirs(os.path.join(data_dir, folder), exist_ok=True)
    with open(os.path.join(data_dir, folder, name), "wb") as f:
        f.write(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    every = sorted(DATASETS) + sorted(OPENML)
    ap.add_argument("--datasets", nargs="+", choices=every, default=every)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--check", action="store_true", help="only check the files; download nothing")
    args = ap.parse_args()

    bad = 0
    for folder in args.datasets:
        if folder in OPENML:
            _, file_id, name, md5, want = OPENML[folder]
            files = {name: want}
            path = os.path.join(args.data_dir, folder, name)
            if not args.check and status(path, want) != "ok":
                fetch_openml(folder, file_id, name, md5, args.data_dir)
            paths = {name: path}
        else:
            uci_id, zip_name, files = DATASETS[folder]
            paths = {name: os.path.join(args.data_dir, folder, name) for name in files}
            if not args.check and any(status(p, files[n]) != "ok" for n, p in paths.items()):
                fetch(folder, uci_id, zip_name, files, args.data_dir)
        for name, path in paths.items():
            s = status(path, files[name])
            bad += s != "ok"
            note = {"ok": "matches the copy our runs used",
                    "missing": "missing",
                    "changed": "DIFFERS from the copy our runs used"}[s]
            print(f"  {folder}/{name}: {note}")
    if bad:
        sys.exit(f"{bad} file(s) missing or different")
    print("all files present and identical to the ones our runs used")


if __name__ == "__main__":
    main()
