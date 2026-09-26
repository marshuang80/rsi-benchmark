# Toy release for `biomedical-tinyclip-data-curation`

Synthetic stand-in for the BIOMEDICA subset, used only to exercise the task end
to end (image builds, baseline, validation, hidden test, calibration) before the
real data is ready. **Not biomedical data**: every figure is drawn procedurally
and its caption is templated from the drawing parameters, so a CLIP model can
learn the alignment and the pool contains realistic noise (non-target figure
types, mismatched or uninformative captions, near-duplicates).

Layout is exactly what `environment/fetch_release.py` expects:

```
pool/{images.bin,index.npy,meta.jsonl}   5,000 pairs   (selection bounds 166..1000, baseline 833)
val/{images.bin,index.npy,meta.jsonl}      400 pairs   (100 per target figure type)
test/{images.bin,index.npy,meta.jsonl}     400 pairs   (100 per target figure type)
pool/labels.jsonl                          hidden labels (reference anchors only; not fetched)
SHA256SUMS, stats.json
```

Regenerate:

```bash
python3 make_toy_shards.py --out shards --shards 60 --per-shard 200
python3 <task>/environment/data_prep/prepare_release.py --out release --local-shard-dir shards \
    --shard-start 0 --shard-stop 60 --shard-step 1 --samples-per-shard 200 \
    --pool-size 5000 --eval-per-type 100 --eval-reserve 0.3
```

Served at `https://raw.githubusercontent.com/marshuang80/rsi-benchmark/toy-data/<path>`.
To switch the task to the real release, change `RELEASE_BASE` and `RELEASE_FILES`
in both copies of `fetch_release.py` and re-measure the anchors in `task.toml`.
