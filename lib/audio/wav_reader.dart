import 'dart:io';
import 'dart:typed_data';

class WavReader {
  static Float32List readMonoPcm16AsFloat32(String path) {
    final bytes = File(path).readAsBytesSync();
    final data = ByteData.sublistView(bytes);

    if (String.fromCharCodes(bytes.sublist(0, 4)) != 'RIFF') {
      throw const FormatException('Not a WAV RIFF file');
    }

    int offset = 12;

    int? audioFormat;
    int? channels;
    int? sampleRate;
    int? bitsPerSample;
    int? dataOffset;
    int? dataSize;

    while (offset < bytes.length) {
      final chunkId = String.fromCharCodes(bytes.sublist(offset, offset + 4));
      final chunkSize = data.getUint32(offset + 4, Endian.little);

      if (chunkId == 'fmt ') {
        audioFormat = data.getUint16(offset + 8, Endian.little);
        channels = data.getUint16(offset + 10, Endian.little);
        sampleRate = data.getUint32(offset + 12, Endian.little);
        bitsPerSample = data.getUint16(offset + 22, Endian.little);
      }

      if (chunkId == 'data') {
        dataOffset = offset + 8;
        dataSize = chunkSize;
        break;
      }

      offset += 8 + chunkSize;
    }

    if (audioFormat != 1) {
      throw FormatException(
        'Only PCM WAV is supported. Got format $audioFormat',
      );
    }

    if (bitsPerSample != 16) {
      throw FormatException(
        'Only 16-bit PCM WAV is supported. Got $bitsPerSample',
      );
    }

    if (sampleRate != 16000) {
      throw FormatException(
        'Expected 16 kHz audio. Got $sampleRate',
      );
    }

    if (dataOffset == null || dataSize == null) {
      throw const FormatException('No WAV data chunk found');
    }

    final channelCount = channels ?? 1;
    final frameCount = dataSize ~/ 2 ~/ channelCount;
    final samples = Float32List(frameCount);

    int byteIndex = dataOffset;

    for (int i = 0; i < frameCount; i++) {
      int sum = 0;

      for (int ch = 0; ch < channelCount; ch++) {
        sum += data.getInt16(byteIndex, Endian.little);
        byteIndex += 2;
      }

      final mono = sum / channelCount;
      samples[i] = mono / 32768.0;
    }

    return samples;
  }
}