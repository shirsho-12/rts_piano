import argparse
import shutil
from pathlib import Path
from typing import Dict

import sentencepiece as spm

from feature_extract import compute_fbank
from prepare_maestro import create_manifest
from preprocess_dataset import preprocess_data_speech


def generate_tokens(lang_dir: Path):
    """
    Generate the tokens.txt from a bpe model.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(lang_dir / "bpe.model"))
    token2id: Dict[str, int] = {sp.id_to_piece(i): i for i in range(sp.vocab_size())}
    with open(lang_dir / "tokens.txt", "w", encoding="utf-8") as f:
        for sym, i in token2id.items():
            f.write(f"{sym} {i}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Maestro Manifest")
    parser.add_argument(
        "--maestro_path", type=str, help="path downloaded Maestro Dataset"
    )
    args = parser.parse_args()

    maestro_path = args.maestro_path
    maestro_dir = Path(maestro_path)

    print("Creating the manifest...")
    create_manifest(maestro_dir)

    print("Preprocessing (removing vel and offset tokens)...")
    preprocess_data_speech(
        maestro_dir,
        maestro_dir,
        dataset_parts=("TRAIN", "TEST", "DEV"),
        prefix="maestro",
        max_dur=60,
        min_dur=1,
    )

    print("Generating the bpe...")
    # Generating the bpe
    user_defined_symbols = ["<blk>", "<sos/eos>"]
    unk_id = len(user_defined_symbols)

    vocab_size = 218
    lang_dir = maestro_dir / "lang"

    model_type = "word"

    model_prefix = f"{lang_dir}/{model_type}_{vocab_size}"
    train_text = "model/words.txt"
    character_coverage = 1
    input_sentence_size = 100000000

    # Note: unk_id is fixed to 2.
    # If you change it, you should also change other
    # places that are using it.

    model_file = Path(model_prefix + ".model")
    if not model_file.is_file():
        spm.SentencePieceTrainer.train(
            input=train_text,
            vocab_size=vocab_size,
            model_type=model_type,
            model_prefix=model_prefix,
            input_sentence_size=input_sentence_size,
            character_coverage=character_coverage,
            user_defined_symbols=user_defined_symbols,
            unk_id=unk_id,
            bos_id=-1,
            eos_id=-1,
        )
    else:
        print(f"{model_file} exists - skipping")

    shutil.copyfile(model_file, f"{lang_dir}/bpe.model")

    generate_tokens(lang_dir)

    print("Generating the features (This will take a while)...")
    compute_fbank(
        src_dir=maestro_dir,
        output_dir=maestro_dir,
        prefix="maestro",
        dataset="TRAIN TEST DEV",
        perturb_speed=False,
    )
