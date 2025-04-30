import logging
from pathlib import Path
from typing import Callable, Optional, Sequence

from lhotse import CutSet
from lhotse.recipes.utils import read_manifests_if_cached


def remove_zero_vel(supervision):
    supervision.text = re.sub(r" <vel:0> <note:[^>]*>", "", supervision.text)
    # removing all velocity as well!
    supervision.text = re.sub(r" <vel:[^>]*>", "", supervision.text)
    return supervision


def preprocess_data_speech(
    src_dir: Path,
    output_dir: Path,
    prefix: str = "",
    suffix: Optional[str] = "jsonl.gz",
    dataset_parts: Optional[Sequence[str]] = None,
    sampling_rate: int = 16000,
    min_dur: float = 1.0,
    max_dur: float = 30.0,
):
    """
    Prepares the CutSet from the supervisions and recording manifest.
    The manifests are searched for using the pattern ``src_dir / f'{prefix}_{manifest}_{part}.{suffix}'``,
    where `manifest` is one of ``["recordings", "supervisions"]`` and ``part`` is specified in ``dataset_parts``.
    If text_field not specified or is "text", CutSet will be saved to ``output_dir / f'{prefix}_cuts_{partition}_raw.{suffix}'``
    else ``output_dir / f'{prefix}_{text_field}_cuts_{partition}_raw.{suffix}'``

    text_filter_{pre/post}_norm: Filter for SupervisionSegment. Will be used via `supervisions.filter(text_filter)`.
                 One filter is for pre-normalization the other for post-normalization.
    normalize_text: Function for normalizing the text.
    text_field: field to use as the training text. Will replace the supervision text with this field. If None or empty will
                use supervision text
    sampling_rate: Rate to sample to.
    merge_supervisions_max_pause: Merge supervisions if the pause between them is less than this.
                                  If value is negative will just trim to supervisions
    min_dur: cut minimum duration
    max_dur: cut maximum duration
    """

    output_dir.mkdir(exist_ok=True)

    if dataset_parts is None:
        dataset_parts = [""]

    logging.info("Loading manifest (may take 4 minutes)")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    for partition, m in manifests.items():
        logging.warning(f"Processing {partition}")
        raw_cuts_path = output_dir / f"{prefix}_cuts_{partition}_raw.{suffix}"
        if raw_cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        # Create long-recording cut manifests.
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"].resample(sampling_rate),
            supervisions=m["supervisions"],
        )

        # removing those cuts without supervisions
        cut_set = (
            cut_set.filter(lambda c: len(c.supervisions) > 0)
            .trim_to_supervisions()
            .filter(lambda c: c.duration >= min_dur and c.duration <= max_dur)
            .filter(lambda c: not c.has_overlapping_supervisions)
            .filter(lambda c: c.supervisions[0].text.strip())
            .filter(lambda cut: cut.supervisions[0].end <= cut.duration + 2e-3)
            .map_supervisions(remove_zero_vel)
        )
        # Run data augmentation that needs to be done in the
        # time domain.
        #  if partition not in ["DEV", "TEST"]:
        #      logging.info(
        #          f"Speed perturb for {partition} with factors 0.9 and 1.1 "
        #          "(Perturbing may take 8 minutes and saving may"
        #          " take 20 minutes)"
        #      )
        #      cut_set = (
        #          cut_set
        #          + cut_set.perturb_speed(0.9)
        #          + cut_set.perturb_speed(1.1)
        #      )
        #
        # Note: No need to perturb the training subset as not all of the
        # data is going to be used in the training.
        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)
