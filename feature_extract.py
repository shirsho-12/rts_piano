import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter

sys.path.insert(-1, "../icefall")
sys.path.insert(-1, "../icefall/egs/librispeech/ASR/local")

# from icefall.utils import get_executor, str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank(
    src_dir: Path = Path("data/manifests"),
    output_dir: Path = Path("data/fbank"),
    prefix: str = "",
    suffix: Optional[str] = "jsonl.gz",
    bpe_model: Optional[str] = None,
    dataset: Optional[str] = None,
    perturb_speed: Optional[bool] = True,
):
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    if bpe_model:
        logging.info(f"Loading {bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    if dataset is None:
        dataset_parts = (
            "TEST",
            "DEV",
            "TRAIN",
        )
    else:
        dataset_parts = dataset.split(" ", -1)

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    # with get_executor() as ex:  # Initialize the executor only once.
    ex = None
    for partition in dataset_parts:
        raw_cuts_path = src_dir / f"{prefix}_cuts_{partition}_raw.{suffix}"
        if not raw_cuts_path.is_file():
            logging.info(f"{partition} does not exists - skipping.")
            continue
        cuts_path = output_dir / f"{prefix}_cuts_{partition}.{suffix}"
        if (cuts_path).is_file():
            logging.info(f"{partition} already exists - skipping.")
            continue
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_file(raw_cuts_path)

        if "TRAIN" in partition:
            if bpe_model:
                cut_set = filter_cuts(cut_set, sp)
            if perturb_speed:
                logging.info(f"Doing speed perturb")
                cut_set = (
                    cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)
                )
        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/{prefix}_feats_{partition}",
            # when an executor is specified, make more partitions
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )
        cut_set.to_file(cuts_path)
