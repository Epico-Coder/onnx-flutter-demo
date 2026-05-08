# ESPnet ONNX Inference Test App

This Flutter app is a small offline speech-recognition test bed for an ESPnet
ASR model exported to ONNX.

The app can:

- Record microphone audio as 16 kHz mono WAV.
- Extract log-mel filterbank features in Dart.
- Run the exported ESPnet ONNX encoder and CTC models.
- Try three decoding modes:
  - Greedy CTC
  - CTC prefix beam search
  - Joint CTC + Transformer beam search
- Save intermediate preprocessing and inference artifacts as CSV files for
  comparison against Python/ESPnet.

The app is intended for debugging and experimentation.

## Current Status

The following parts are working:

- Audio recording
- WAV reading
- Feature extraction
- Encoder ONNX inference
- CTC ONNX inference
- Greedy CTC decoding
- CTC prefix beam decoding
- Intermediate feature dumps

The joint CTC + Transformer decoder path is experimental. The exported
`xformer_decoder.onnx` uses growing decoder cache tensors, and this currently
causes ONNX Runtime shape errors in the Flutter runtime. It is still exposed in
the UI so the behavior can be tested directly.

## Project Structure

Important files:

```text
lib/main.dart
lib/pages/recorder_page.dart
lib/services/recording_service.dart
lib/services/espnet_asr_service.dart
lib/audio/wav_reader.dart
lib/audio/log_mel_extractor.dart
lib/decoding/ctc_decoder.dart
lib/decoding/ctc_prefix_beam_search.dart
lib/decoding/joint_ctc_transformer_beam_search.dart
lib/debug/intermediate_feature_writer.dart
assets/models/espnet_onnx/
```

The ONNX assets are expected here:

```text
assets/models/espnet_onnx/bpe.model
assets/models/espnet_onnx/config.yaml
assets/models/espnet_onnx/default_encoder.onnx
assets/models/espnet_onnx/default_encoder.onnx.data
assets/models/espnet_onnx/ctc.onnx
assets/models/espnet_onnx/ctc.onnx.data
assets/models/espnet_onnx/xformer_decoder.onnx
assets/models/espnet_onnx/xformer_decoder.onnx.data
```

These files are listed in `pubspec.yaml` as Flutter assets.

## Requirements

- Flutter SDK installed
- Dart SDK included with Flutter
- A supported desktop/mobile target
- Microphone access

Check your Flutter install:

```sh
flutter doctor
```

List available devices:

```sh
flutter devices
```

## Installation

From this directory:

```sh
flutter pub get
```

Run the app:

```sh
flutter run
```

For a specific device:

```sh
flutter run -d <device_id>
```

## Usage

1. Start the app.
2. Allow microphone permission if prompted.
3. Select a microphone.
4. Select a decoding mode:
   - `Greedy CTC`: fastest and simplest, but usually weakest.
   - `CTC Prefix Beam`: stronger CTC-only decoding and currently the most
     stable mode.
   - `Joint CTC + Transformer`: experimental; may fail with ONNX cache-shape
     errors.
5. Press `Record`.
6. Speak.
7. Press `Stop Record`.
8. Press `Run ONNX Test`.
9. Read the transcript or error message.

## Output Files

The app writes files to Flutter's app documents directory, not to the project
folder.

On each machine/platform, Flutter chooses the correct writable app data
location. The app creates these folders automatically:

```text
<app documents>/recordings/
<app documents>/intermediate_features/
<app documents>/espnet_onnx/
```

`recordings/` contains recorded WAV files.

`intermediate_features/` contains one timestamped folder per transcription run.
Each run folder may contain:

```text
resampled_waveform.csv
padded_waveform.csv
frames.csv
fft_window.csv
power_spectrum.csv
mel_basis.csv
mel_output.csv
logmel.csv
per_bin_mean_std.csv
normalized_features.csv
encoder_output.csv
ctc_logits_or_log_probs.csv
token_ids.csv
metadata.txt
```

`espnet_onnx/` contains copied model files. The model assets are copied there on
first initialization because ONNX Runtime needs filesystem paths.

The app prints the intermediate feature folder path with `debugPrint`.

## Feature Extraction

The Dart frontend uses the parameters from the ESPnet ONNX config:

```text
sample rate: 16000
n_fft: 512
win_length: 400
hop_length: 160
n_mels: 80
center: true
window: Hann
mel scale: Slaney-style, htk=false
normalization: utterance mean normalization
```

The app currently expects 16 kHz WAV input. It does not resample arbitrary audio
files. Recordings made by the app are requested as 16 kHz mono WAV.
