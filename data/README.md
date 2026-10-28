# Datasets

The paper's four UCI datasets are **not stored in this repository**. Download each once from the UCI
Machine Learning Repository and put the listed files into the matching folder here. The loaders in
`pace_cfmnn/data.py` stop with a clear error that names any missing file.

| Folder | UCI dataset | Files to put in the folder | Rows | Features | Classes |
|---|---|---|---|---|---|
| `thyroid/` | [Thyroid Disease](https://archive.ics.uci.edu/dataset/102/thyroid+disease) | `ann-train.data`, `ann-test.data` | 7,200 | 21 | 3 |
| `shuttle/` | [Statlog (Shuttle)](https://archive.ics.uci.edu/dataset/148/statlog+shuttle) | `shuttle.trn`, `shuttle.tst` | 58,000 | 9 | 7 |
| `skin/` | [Skin Segmentation](https://archive.ics.uci.edu/dataset/229/skin+segmentation) | `Skin_NonSkin.txt` | 245,057 | 3 | 2 |

Wine, Breast Cancer and Digits come with scikit-learn, and S1–S3 are generated, so they need no
download.

## Steps

1. **Thyroid:** download
   [thyroid+disease.zip](https://archive.ics.uci.edu/static/public/102/thyroid+disease.zip) and unzip
   it. Copy `ann-train.data` and `ann-test.data` into `thyroid/`. The zip holds many other thyroid
   files; only this "ann" pair is used.
2. **Shuttle:** download
   [statlog+shuttle.zip](https://archive.ics.uci.edu/static/public/148/statlog+shuttle.zip) and unzip
   it.
   - The training file comes compressed as `shuttle.trn.Z`. Decompress it with `gzip -d shuttle.trn.Z`
     (or 7-Zip on Windows).
   - Copy `shuttle.trn` and `shuttle.tst` into `shuttle/`.
3. **Skin:** download
   [skin+segmentation.zip](https://archive.ics.uci.edu/static/public/229/skin+segmentation.zip), unzip
   it and copy `Skin_NonSkin.txt` into `skin/`.

Check the files with:

```bash
python -m unittest tests.test_data -v
```

`test_uci_sets_match_paper_table_1` compares row, feature and class counts with the paper's Table 1. It
skips any dataset that is not there.

## Notes

- **Skin:** about 79% of rows repeat a colour value, but no colour value appears with both labels.

## Sources

The datasets come from the UCI Machine Learning Repository. Each dataset's page there gives its license
and how to cite it.
