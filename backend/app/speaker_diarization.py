"""Speaker diarization: assign "Speaker 1..N" labels to transcript segments.

Local and offline. Voice embeddings come from Resemblyzer (small pretrained
encoder, CPU) — each segment's audio is embedded and segments are clustered
by voice similarity. Segments shorter than ~0.3 s inherit the previous
speaker, so pauses and breaths never create phantom speakers.

Pipeline position: after transcription, before translation — the labels feed
the Characters system, where each speaker gets a persistent voice.

Gender detection rides on the same pass: median fundamental frequency (F0)
per speaker cluster classifies Female (~>160 Hz) vs Male. The gender feeds
the automatic voice assignment in the Characters profiles.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .providers import get_storage_dir

logger = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD = 0.70
_MIN_SPEECH_SECONDS = 0.3
# Median F0 split between typical adult female and male speech.
_GENDER_F0_THRESHOLD_HZ = 160.0
# Pitch is analyzed on at most this many windows per speaker — the median
# converges long before, and long videos stay fast.
_MAX_PITCH_WINDOWS_PER_SPEAKER = 25
_MIN_PITCH_FRAMES = 10


def is_available() -> bool:
    import importlib.util

    return importlib.util.find_spec('resemblyzer') is not None


def _extract_audio(media_path: str, out_wav: Path, ffmpeg: str) -> None:
    import subprocess

    subprocess.run(
        [ffmpeg, '-y', '-i', media_path, '-vn', '-ac', '1', '-ar', '16000',
         '-c:a', 'pcm_s16le', str(out_wav)],
        check=True, capture_output=True, text=True,
    )


def assign_speakers(media_path: str, segments: list[dict[str, Any]], ffmpeg: str,
                    progress_cb=None) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Label every segment with a speaker name using sliding-window clustering.

    Embeds 2-second windows (1 s hop) across the whole audio, clusters them by
    voice similarity, then gives each Whisper segment the majority label of
    the windows overlapping it. Works even when Whisper merges several
    sentences into one segment. Speakers are numbered by first speech.

    Returns (labels, genders) where genders maps each final speaker name to
    {'gender': 'Female' | 'Male' | None, 'median_f0_hz': float | None}.
    """
    import numpy as np
    from resemblyzer import VoiceEncoder

    from .providers import get_storage_dir

    if not segments:
        return [], {}

    wav_path = get_storage_dir('data') / 'diarize-temp.wav'
    _extract_audio(media_path, wav_path, ffmpeg)

    try:
        import soundfile as sf
        from resemblyzer import preprocess_wav

        audio, sample_rate = sf.read(str(wav_path), dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        encoder = VoiceEncoder('cpu')

        window_sec, hop_sec = 2.0, 1.0
        window_len = int(window_sec * sample_rate)
        hop_len = int(hop_sec * sample_rate)
        window_centers: list[float] = []
        window_vectors: list[np.ndarray] = []
        window_pitches: list[np.ndarray | None] = []

        positions = list(range(0, max(1, len(audio) - int(0.5 * sample_rate) + 1), hop_len))
        for processed, pos in enumerate(positions):
            clip = audio[pos:pos + window_len]
            if len(clip) < int(0.5 * sample_rate):
                continue
            # VAD-trim + normalize INSIDE the window: the window's position is
            # preserved, but silences don't dilute the voice embedding.
            voiced = preprocess_wav(clip, source_sr=sample_rate)
            if len(voiced) < int(0.5 * sample_rate):
                continue  # window is mostly silence
            window_centers.append((pos + window_len / 2) / sample_rate)
            window_vectors.append(encoder.embed_utterance(voiced))
            window_pitches.append(_pitch_track(voiced, sample_rate))
            if progress_cb:
                progress_cb(min(99, int(90 * (processed + 1) / len(positions))))

        # Greedy clustering over window embeddings.
        cluster_names: list[str] = []
        cluster_vectors: list[np.ndarray] = []
        cluster_pitches: dict[int, list[np.ndarray]] = {}
        window_labels: list[str] = []
        for index, vector in enumerate(window_vectors):
            best_cluster, best_similarity = -1, -1.0
            for cluster, centroid in enumerate(cluster_vectors):
                similarity = float(np.dot(vector, centroid) /
                                   (np.linalg.norm(vector) * np.linalg.norm(centroid) + 1e-9))
                if similarity > best_similarity:
                    best_cluster, best_similarity = cluster, similarity
            if best_cluster < 0 or best_similarity < _SIMILARITY_THRESHOLD:
                cluster_vectors.append(vector.copy())
                cluster_names.append(f'Speaker {len(cluster_names) + 1}')
                best_cluster = len(cluster_names) - 1
            else:
                cluster_vectors[best_cluster] = cluster_vectors[best_cluster] * 0.8 + vector * 0.2
            pitch = window_pitches[index]
            if pitch is not None and len(cluster_pitches.get(best_cluster, [])) < _MAX_PITCH_WINDOWS_PER_SPEAKER:
                cluster_pitches.setdefault(best_cluster, []).append(pitch)
            window_labels.append(cluster_names[best_cluster])

        # Majority vote per Whisper segment over overlapping windows.
        labels: list[str] = []
        for segment in segments:
            start = float(segment.get('start') or 0)
            end = float(segment.get('end') or start)
            votes: list[str] = []
            for center, label in zip(window_centers, window_labels):
                if start - hop_sec <= center <= end + hop_sec:
                    votes.append(label)
            if votes:
                labels.append(max(set(votes), key=votes.count))
            elif window_labels:
                labels.append(window_labels[0])
            else:
                labels.append('Speaker 1')
        # Renumber by first appearance so speakers are Speaker 1, 2, ... in
        # dialogue order (transitional windows may leave gaps in the raw IDs).
        order: list[str] = []
        for label in labels:
            if label not in order:
                order.append(label)
        renamed = {old_name: f'Speaker {position + 1}' for position, old_name in enumerate(order)}

        genders: dict[str, dict[str, Any]] = {}
        for old_name, new_name in renamed.items():
            cluster_index = cluster_names.index(old_name)
            pitches = [frame for group in cluster_pitches.get(cluster_index, []) for frame in group]
            genders[new_name] = _classify_gender(pitches)
        return [renamed[label] for label in labels], genders
    finally:
        wav_path.unlink(missing_ok=True)


def _pitch_track(samples: Any, sample_rate: int) -> np.ndarray | None:
    """Fundamental-frequency frames (Hz) of a voiced clip, unvoiced frames dropped."""
    try:
        import librosa
        import numpy as np

        f0, _, _ = librosa.pyin(
            np.asarray(samples, dtype=np.float32),
            fmin=65.0, fmax=350.0, sr=sample_rate, frame_length=1024,
        )
        if f0 is None:
            return None
        return f0[~np.isnan(f0)]
    except Exception as error:  # noqa: BLE001 - a window without usable pitch is not fatal
        logger.warning('Pitch analysis failed on one window: %s', error)
        return None


def _classify_gender(pitches: list[Any]) -> dict[str, Any]:
    import numpy as np

    if len(pitches) < _MIN_PITCH_FRAMES:
        return {'gender': None, 'median_f0_hz': None}
    median_f0 = float(np.median(np.asarray(pitches, dtype=np.float64)))
    gender = 'Female' if median_f0 >= _GENDER_F0_THRESHOLD_HZ else 'Male'
    return {'gender': gender, 'median_f0_hz': round(median_f0, 1)}