# Running the demo

The requirement packages for the demo is listed in `requirements_deployment.txt` and can be installed via the following command:

```bash
# Creating the conda env
conda create --name live_piano_trans python==3.10
conda activate live_piano_trans
pip install -r requirements_deployment.txt
```

To start the live demo, run:

```bash
python visualiser.py --tokens model/tokens.txt --decoder model/decoder-epoch-17-avg-6-chunk-8-left-128.onnx --encoder model/encoder-epoch-17-avg-6-chunk-8-left-128.onnx --joiner model/joiner-epoch-17-avg-6-chunk-8-left-128.onnx
```

# Training the model

You can download the dataset from: https://magenta.tensorflow.org/datasets/maestro#v300

The following package are required to be installed and you need to follow the link for the instruction for installation.

- torch, torchaudio (https://pytorch.org/get-started/locally/)
- kaldifeat (https://csukuangfj.github.io/kaldifeat/installation/from_wheels.html)
- k2 (https://k2-fsa.github.io/k2/installation/from_wheels.html)

For `k2` and `kaldifeat` the main thing to ensure is that the torch and cuda version matches that of the one that is installed.

The remaining package requirement can be installed from `requirement_training.txt` via:

```bash
pip install requirements_training.txt
```

## Preprocessing: Creating Manifest, CutSet, Training BPE and feature extraction

Run

```bash
python run_create_manifest.py --maestro-path <PATH_TO_MAESTRO>
```

This will create the manifest and feature in the Maestro directory.

## Starting the training

For this we require the additional icefall scripts:

```bash
git clone https://github.com/k2-fsa/icefall
cd icefall
# Freezing at current commit to prevent breaking changes
git checkout 57451b03828bf5325c91eaf12c80aaae9063283d
# Adding the icefall folder to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}:${PWD}/egs/librispeech/ASR/zipformer/"
```

The training can be run via:

```bash
## Training the first 2 epoch without delay penalty
python ./zipformer_mod/train.py \
    --bpe-model <PATH_TO_MAESTRO>/lang/bpe.model \
    --exp-dir <OUTPUT_MODEL_DIR> \
    --manifest-dir <PATH_TO_MAESTRO> \
    --use-fp16 1 \
    --base-lr 0.04 \
    --enable-musan False \
    --enable-spec-aug False \
    --tensorboard False \
    --max-duration 400 \
    --use-cr-ctc 0 \
    --use-ctc 0 \
    --use-transducer 1 \
    --use-attention-decoder 0 \
    --num-encoder-layers 2,2,2,2,2 \
    --feedforward-dim 768,768,768,768,768 \
    --num-heads 8,8,8,8,8 \
    --encoder-dim 256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192 \
    --downsampling-factor 1,2,4,8,2 \
    --cnn-module-kernel 31,31,15,31,31 \
    --decoder-dim 512 \
    --joiner-dim 512 \
    --chunk-size 8,16,32 \
    --left-context-frames 128,256 \
    --num-epochs 2 \
    --causal 1 \
    --concatenate-cuts False \
    --prefix maestro \
    --small-dev True

## Adding the delay penalty for subsequent epoch
python ./zipformer_mod/train.py \
    --bpe-model <PATH_TO_MAESTRO>/lang/bpe.model \
    --exp-dir <OUTPUT_MODEL_DIR> \
    --manifest-dir <PATH_TO_MAESTRO> \
    --use-fp16 1 \
    --base-lr 0.04 \
    --enable-musan False \
    --enable-spec-aug False \
    --tensorboard False \
    --max-duration 400 \
    --use-cr-ctc 0 \
    --use-ctc 0 \
    --use-transducer 1 \
    --use-attention-decoder 0 \
    --num-encoder-layers 2,2,2,2,2 \
    --feedforward-dim 768,768,768,768,768 \
    --num-heads 8,8,8,8,8 \
    --encoder-dim 256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192 \
    --downsampling-factor 1,2,4,8,2 \
    --cnn-module-kernel 31,31,15,31,31 \
    --decoder-dim 512 \
    --joiner-dim 512 \
     --delay-penalty 0.008 \
    --chunk-size 8,16,32 \
    --left-context-frames 128,256 \
    --start-epoch 3 \
    --num-epochs 20 \
    --causal 1 \
    --concatenate-cuts False \
    --prefix maestro \
    --small-dev True
```

## Converting to onnx

This requires the following additional packages:

```bash
pip install onnxruntime onnx onnxconverter_common
```

Using the exporter from k2-icefall:

```bash
python <ICEFALL_FOLDER>/egs/librispeech/ASR/zipformer/export-onnx-streaming.py \
  --tokens <PATH_TO_MAESTRO>/lang/tokens.txt \
  --use-averaged-model 1 \
  --epoch 17 \
  --avg 6 \
  --exp-dir <OUTPUT_MODEL_DIR> \
  --num-encoder-layers "2,2,2,2,2" \
  --downsampling-factor "1,2,4,8,2" \
  --feedforward-dim "768,768,768,768,768" \
  --num-heads "8,8,8,8,8" \
  --encoder-dim "256,256,256,256,256" \
  --query-head-dim 32 \
  --value-head-dim 12 \
  --pos-head-dim 4 \
  --pos-dim 48 \
  --encoder-unmasked-dim "192,192,192,192,192" \
  --cnn-module-kernel "31,31,15,31,31" \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --causal True \
  --chunk-size 8 \
  --left-context-frames 128 \
  --fp16 True
```
