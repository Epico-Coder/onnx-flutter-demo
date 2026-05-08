import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

class RecordingService {
  final AudioRecorder _recorder = AudioRecorder();

  Future<List<InputDevice>> listInputDevices() async {
    final hasPermission = await _recorder.hasPermission();

    if (!hasPermission) {
      throw Exception('Microphone permission denied');
    }

    return _recorder.listInputDevices();
  }

  Future<String> start({InputDevice? device}) async {
    final hasPermission = await _recorder.hasPermission();

    if (!hasPermission) {
      throw Exception('Microphone permission denied');
    }

    final path = await _createRecordingPath();

    await _recorder.start(
      RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
        device: device,
      ),
      path: path,
    );

    return path;
  }

  Future<String?> stop() async {
    return _recorder.stop();
  }

  Future<String> _createRecordingPath() async {
    final documents = await getApplicationDocumentsDirectory();
    final recordingsDir = Directory('${documents.path}/recordings');

    if (!await recordingsDir.exists()) {
      await recordingsDir.create(recursive: true);
    }

    final timestamp = DateTime.now().millisecondsSinceEpoch;
    return '${recordingsDir.path}/recording_$timestamp.wav';
  }

  Future<void> dispose() async {
    await _recorder.dispose();
  }
}
