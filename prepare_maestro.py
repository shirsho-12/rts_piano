import json
from pathlib import Path

import tqdm
from lhotse import RecordingSet, SupervisionSegment, SupervisionSet

from midi_utils import convert_midi_event, get_suit_timing


def create_recordings(maestro_dir: Path, output_dir: Path):
    maestro_wavs = RecordingSet.from_dir(maestro_dir, pattern="*/*.wav", num_jobs=2)
    maestro_wavs.to_file(output_dir / "Maestro_recordings.jsonl.gz")


def create_manifest(maestro_dir: Path, duration_cap: float = 25):
    maestro_json = maestro_dir / "maestro-v3.0.0.json"
    with maestro_json.open("r") as f:
        maestro_dict = json.load(f)

    split_dict = {
        "test": [],
        "train": [],
        "validation": [],
    }
    sup_ls = []

    maestro_recordings = RecordingSet.from_dir(
        maestro_dir, pattern="*/*.wav", num_jobs=2
    )
    for key, split in tqdm(maestro_dict["split"].items()):
        recording_id = (
            maestro_dict["audio_filename"][key].split("/")[-1].replace(".wav", "")
        )
        split_dict[split].append(recording_id)
        midi_file = maestro_dir / maestro_dict["midi_filename"][key]
        midi_events = convert_midi_event(midi_file)
        suit_list = get_suit_timing(midi_events)
        seg_id = 0
        text = ""
        start_time = None

        for suit in suit_list:
            if start_time is None:
                start_time = suit["start"]
            end_time = suit["end"]
            text += suit["stm"]
            if end_time - start_time >= duration_cap:
                sup = SupervisionSegment(
                    id=f"{recording_id}_{seg_id:05d}",
                    recording_id=recording_id,
                    start=start_time,
                    duration=end_time - start_time,
                    text=text,
                )
                sup_ls.append(sup)
                seg_id += 1
                start_time = None
                text = ""
        if start_time is not None:
            sup = SupervisionSegment(
                id=f"{recording_id}_{seg_id:05d}",
                recording_id=recording_id,
                start=start_time,
                duration=end_time - start_time,
                text=text,
            )
            sup_ls.append(sup)

    maestro_supervisions = SupervisionSet.from_segments(sup_ls)

    # splitting the data to splits
    mapping = {
        "train": "TRAIN",
        "test": "TEST",
        "validation": "DEV",
    }
    for split in ["train", "test", "validation"]:
        recs = maestro_recordings.filter(lambda x: x.id in split_dict[split])
        recs.to_file(maestro_dir / f"maestro_recordings_{mapping[split]}.jsonl.gz")
        sups = maestro_supervisions.filter(
            lambda x: x.recording_id in split_dict[split]
        )
        sups.to_file(maestro_dir / f"maestro_supervisions_{mapping[split]}.jsonl.gz")
