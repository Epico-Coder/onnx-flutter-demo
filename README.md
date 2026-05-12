# ESPnet ONNX Inference Test App

A small Flutter app that runs an ESPnet-exported Dutch ASR model fully
on-device via ONNX Runtime. Built as a test bed: every step of the pipeline
is auditable, three decoding paths are exposed in the UI, and intermediate
tensors can be dumped to CSV for comparison against a reference.

The app can:

- Record microphone audio as 16 kHz mono WAV.
- Extract log-mel filterbank features in Dart (matches `librosa` to
  ~`1e-3`).
- Run the exported ESPnet ONNX encoder, CTC head, and transformer decoder.
- Decode in three modes:
  - **Greedy CTC** — argmax + repeat/blank collapse.
  - **CTC prefix beam** — full prefix beam search over the CTC posteriors.
  - **Joint CTC + Transformer** — espnet-style joint beam search; the CTC
    prefix score is mixed with an autoregressive transformer decoder.
- Dump intermediate features and tensors as CSV.

## Requirements

- Flutter SDK
- A supported desktop/mobile target (macOS desktop is the primary target
  used during development)
- Microphone access

```sh
flutter doctor
flutter devices
```

## Run

```sh
flutter pub get
flutter run -d macos          # or -d <device_id>
```

In the app:

1. Allow microphone permission if prompted.
2. Pick a microphone.
3. Pick a decoding mode.
4. **Record** → speak → **Stop Record** → **Run ONNX Test**.
5. The transcript appears in the UI; intermediate dumps land in the app's
   documents directory.

## Assets

The ONNX assets are loaded from `assets/models/espnet_onnx/`:

```text
bpe.model
config.yaml
default_encoder.onnx        ← single-file ONNX (no .data sidecar)
ctc.onnx
xformer_decoder.onnx
```

The bundled model is `DutchCGNConformerFBankExport`: 80-dim FBank input,
Conformer encoder, CTC + transformer decoder heads, 5000-token BPE
vocabulary (`<sos/eos>` = 4999, `<blank>` = 0).

On first launch the app copies these assets into the app documents
directory because ONNX Runtime needs filesystem paths. On every later
launch, files in that directory are refreshed when their on-disk size
differs from the bundled asset, and any orphan files (e.g. leftover
`*.onnx.data` sidecars from an older export) are deleted. This avoids the
case where ONNX Runtime silently loads stale external-data weights.

## Decoding paths

| Mode | Algorithm | Notes |
|---|---|---|
| Greedy CTC | per-frame argmax + collapse | `lib/decoding/ctc_decoder.dart` |
| CTC prefix beam | log-softmax once, size-`k` min-heap top-K per frame, linked-list prefix with incremental hash | `lib/decoding/ctc_prefix_beam_search.dart` |
| Joint CTC + Transformer | per-hypothesis KV caches threaded across steps; CTC prefix score (weight 0.3) mixed with cumulative transformer log-probs (weight 0.7); espnet `end_detect` stop condition | `lib/decoding/joint_ctc_transformer_beam_search.dart` (algorithm) + `lib/decoding/transformer_decoder_runner.dart` (ORT plumbing) |

The transformer decoder is fed the full prefix as `tgt` each step and
threads its own `cache_0..cache_5` tensors per hypothesis (initial cache
length 0, grows by 1 per step).

## Tests

```sh
flutter test
```

What's covered (`test/`):

- **`ctc_decoding_test.dart`** — both CTC decoders run on saved CTC
  log-probs (`test/ctc_log_probs.bin`) and assert the exact token list
  matches the Python reference (`"de plant van de aardappel is giftig"`).
- **`log_mel_parity_test.dart`** — Dart `LogMelExtractor` is compared
  value-for-value against `librosa` output (saved as
  `test/log_mel.bin`); current mean/max error is ~0.
- **`joint_decoder_test.dart`** — beam-search algorithm exercised with a
  fake `TransformerDecoderRunner`. Asserts (a) the decoder is called with
  the full prefix, (b) caches are threaded across steps and inherited
  correctly when the beam splits.
- **`ctc_perf_test.dart`** — timing harness for the CTC decoders.

The fake-runner approach for `joint_decoder_test.dart` exists because
`flutter_onnxruntime` is a platform plugin and isn't loadable under
`flutter test`. End-to-end transformer decoding is exercised on-device
via `flutter run` and the in-app test wav (`assets/test.wav`).

## Feature extraction details

Parameters come from `config.yaml`:

```text
sample rate: 16000
n_fft: 512
win_length: 400
hop_length: 160
n_mels: 80
center: true
pad mode: reflect
window: Hann
mel scale: Slaney-style (htk=false)
normalization: utterance mean (norm_means=true, norm_vars=false)
```

WAV input must be 16 kHz mono PCM16. The reader handles both `WAVE_FORMAT_PCM`
and `WAVE_FORMAT_EXTENSIBLE` containers.

## Output files

```text
<app documents>/recordings/
<app documents>/intermediate_features/<timestamp>_<wav>/
<app documents>/espnet_onnx/
```

Each transcription writes a fresh `intermediate_features/<timestamp>_…/`
folder containing CSVs of every intermediate tensor — useful for
side-by-side comparison with a Python reference. The path is logged with
`debugPrint`.

## Re-exporting the model

If you re-export the ESPnet model via `espnet_onnx.export.ASRModelExport`,
two patches to `site-packages/espnet_onnx/export/asr/models/decoders/xformer.py`
are required for the decoder ONNX to work autoregressively:

1. `get_dummy_inputs`: use `tgt` of length 2 and cache of length 1, so the
   tracer doesn't fold the `x_q = x[:, -1:, :]` slice into a no-op (which
   happens when both lengths are 1 at trace time).

2. `forward`: drop the inter-layer `x = x[:, 1:, :]` strip. It mismatches
   the attention mask in later layers and makes the model only run
   correctly when `tgt_length == 1`.

After re-exporting, place the new files in `assets/models/espnet_onnx/`.
The app's first launch will refresh its cached copies automatically.

Verify a fresh export with a multi-step run before shipping it (cache
should grow by 1 each step, `y` shape `(1, 5000)`):

```py
import numpy as np, onnxruntime as ort
dec = ort.InferenceSession("xformer_decoder.onnx")
mem = np.random.randn(1, 100, 256).astype(np.float32)
caches = [np.zeros((1, 0, 256), dtype=np.float32) for _ in range(6)]
for step, prefix in enumerate([[4999], [4999, 100], [4999, 100, 200]]):
    out = dec.run(None, {"tgt": np.array([prefix], dtype=np.int64),
                         "memory": mem,
                         **{f"cache_{i}": c for i, c in enumerate(caches)}})
    caches = list(out[1:])
    print(f"step {step}: y={out[0].shape}, cache len={caches[0].shape[1]}")
```

## Project structure

```text
lib/main.dart
lib/pages/recorder_page.dart
lib/services/recording_service.dart
lib/services/espnet_asr_service.dart       ← orchestrates the pipeline
lib/audio/wav_reader.dart
lib/audio/log_mel_extractor.dart
lib/decoding/ctc_decoder.dart              ← greedy
lib/decoding/ctc_prefix_beam_search.dart   ← prefix beam
lib/decoding/ctc_logits.dart               ← shared log-softmax / top-K
lib/decoding/joint_ctc_transformer_beam_search.dart
lib/decoding/transformer_decoder_runner.dart
lib/debug/intermediate_feature_writer.dart
assets/models/espnet_onnx/
assets/test.wav                            ← reference audio
test/
```
