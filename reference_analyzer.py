#!/usr/bin/env python3
"""
🎯 REFERENCE ANALYZER - PROFESSIONAL MASTERING STANDARDS
Analyzes user reference tracks to extract professional mastering characteristics

Features:
- LUFS, dynamic range, frequency balance analysis
- Mastering standard extraction from professional tracks
- Reference-based target optimization
- Electronic music genre detection and classification
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf


class ReferenceAnalyzer:
    """Analyzes reference tracks to extract professional mastering standards"""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(sample_rate)

    def analyze_reference_track(self, audio_path: str) -> Dict:
        """Analyze a single reference track for mastering characteristics"""
        print(f"📊 Analyzing: {os.path.basename(audio_path)}")

        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)

            # Convert to mono for some analyses
            mono_audio = np.mean(audio, axis=0) if audio.ndim > 1 else audio

            # Basic measurements
            peak = np.max(np.abs(audio))
            rms = np.sqrt(np.mean(audio**2))

            # LUFS measurement
            try:
                lufs = self.meter.integrated_loudness(audio.T)
            except:
                lufs = -23.0  # Fallback

            # Dynamic range (crest factor)
            dynamic_range = 20 * np.log10(peak / (rms + 1e-10))

            # Frequency analysis
            fft = np.fft.rfft(mono_audio)
            freqs = np.fft.rfftfreq(len(mono_audio), 1 / sr)
            magnitude = np.abs(fft)

            # Frequency band analysis
            bass_mask = (freqs >= 20) & (freqs <= 250)
            mid_mask = (freqs >= 250) & (freqs <= 4000)
            high_mask = (freqs >= 4000) & (freqs <= 20000)

            bass_energy = np.mean(magnitude[bass_mask]) if np.any(bass_mask) else 0
            mid_energy = np.mean(magnitude[mid_mask]) if np.any(mid_mask) else 0
            high_energy = np.mean(magnitude[high_mask]) if np.any(high_mask) else 0

            total_energy = bass_energy + mid_energy + high_energy
            if total_energy > 0:
                bass_ratio = bass_energy / total_energy
                mid_ratio = mid_energy / total_energy
                high_ratio = high_energy / total_energy
            else:
                bass_ratio = mid_ratio = high_ratio = 0.33

            # Stereo characteristics
            if audio.ndim > 1 and audio.shape[0] == 2:
                correlation = np.corrcoef(audio[0], audio[1])[0, 1]
                stereo_width = 1.0 - abs(correlation)
            else:
                stereo_width = 0.0

            # Spectral characteristics
            spectral_centroid = librosa.feature.spectral_centroid(y=mono_audio, sr=sr)[
                0
            ]
            avg_centroid = np.mean(spectral_centroid)

            spectral_rolloff = librosa.feature.spectral_rolloff(y=mono_audio, sr=sr)[0]
            avg_rolloff = np.mean(spectral_rolloff)

            # Onset detection for rhythm analysis
            onset_frames = librosa.onset.onset_detect(y=mono_audio, sr=sr)
            onset_density = len(onset_frames) / (
                len(mono_audio) / sr
            )  # onsets per second

            analysis = {
                "filename": os.path.basename(audio_path),
                "peak": float(peak),
                "rms": float(rms),
                "lufs": float(lufs),
                "dynamic_range": float(dynamic_range),
                "bass_ratio": float(bass_ratio),
                "mid_ratio": float(mid_ratio),
                "high_ratio": float(high_ratio),
                "stereo_width": float(stereo_width),
                "spectral_centroid": float(avg_centroid),
                "spectral_rolloff": float(avg_rolloff),
                "onset_density": float(onset_density),
                "duration": float(len(mono_audio) / sr),
                "sample_rate": sr,
            }

            # Classify genre/style based on characteristics
            analysis["style"] = self._classify_style(analysis)

            print(f"  ✅ LUFS: {lufs:.1f}, Peak: {peak:.3f}, DR: {dynamic_range:.1f}dB")
            return analysis

        except Exception as e:
            print(f"  ❌ Error analyzing {audio_path}: {e}")
            return None

    def _classify_style(self, analysis: Dict) -> str:
        """Classify the musical style based on analysis"""
        lufs = analysis["lufs"]
        bass_ratio = analysis["bass_ratio"]
        onset_density = analysis["onset_density"]
        stereo_width = analysis["stereo_width"]

        # Electronic music classification
        if lufs > -8 and bass_ratio > 0.4:
            if onset_density > 2:
                return "club_techno"
            else:
                return "electronic_ambient"
        elif lufs > -12 and bass_ratio > 0.35:
            return "electronic_house"
        elif stereo_width > 0.6:
            return "electronic_experimental"
        else:
            return "electronic_general"

    def analyze_reference_collection(self, references_dir: str) -> Dict:
        """Analyze entire reference collection and extract mastering standards"""
        print(f"🎯 ANALYZING REFERENCE COLLECTION: {references_dir}")
        print("=" * 60)

        if not os.path.exists(references_dir):
            print(f"❌ References directory not found: {references_dir}")
            return {}

        analyses = []
        wav_files = [
            f for f in os.listdir(references_dir) if f.lower().endswith(".wav")
        ]

        print(f"Found {len(wav_files)} reference tracks")

        for wav_file in wav_files:
            audio_path = os.path.join(references_dir, wav_file)
            analysis = self.analyze_reference_track(audio_path)
            if analysis:
                analyses.append(analysis)

        if not analyses:
            print("❌ No valid references found")
            return {}

        # Calculate aggregate statistics
        print(f"\n📈 REFERENCE STATISTICS ({len(analyses)} tracks):")
        print("-" * 40)

        lufs_values = [a["lufs"] for a in analyses]
        peak_values = [a["peak"] for a in analyses]
        dr_values = [a["dynamic_range"] for a in analyses]
        bass_ratios = [a["bass_ratio"] for a in analyses]
        high_ratios = [a["high_ratio"] for a in analyses]

        stats = {
            "num_tracks": len(analyses),
            "lufs": {
                "mean": float(np.mean(lufs_values)),
                "std": float(np.std(lufs_values)),
                "min": float(np.min(lufs_values)),
                "max": float(np.max(lufs_values)),
                "median": float(np.median(lufs_values)),
            },
            "peak": {
                "mean": float(np.mean(peak_values)),
                "std": float(np.std(peak_values)),
                "min": float(np.min(peak_values)),
                "max": float(np.max(peak_values)),
                "median": float(np.median(peak_values)),
            },
            "dynamic_range": {
                "mean": float(np.mean(dr_values)),
                "std": float(np.std(dr_values)),
                "min": float(np.min(dr_values)),
                "max": float(np.max(dr_values)),
                "median": float(np.median(dr_values)),
            },
            "bass_ratio": {
                "mean": float(np.mean(bass_ratios)),
                "std": float(np.std(bass_ratios)),
                "min": float(np.min(bass_ratios)),
                "max": float(np.max(bass_ratios)),
                "median": float(np.median(bass_ratios)),
            },
            "high_ratio": {
                "mean": float(np.mean(high_ratios)),
                "std": float(np.std(high_ratios)),
                "min": float(np.min(high_ratios)),
                "max": float(np.max(high_ratios)),
                "median": float(np.median(high_ratios)),
            },
            "detailed_analyses": analyses,
        }

        # Print summary
        print(
            f"LUFS Range: {stats['lufs']['min']:.1f} to {stats['lufs']['max']:.1f} (avg: {stats['lufs']['mean']:.1f})"
        )
        print(
            f"Peak Range: {stats['peak']['min']:.3f} to {stats['peak']['max']:.3f} (avg: {stats['peak']['mean']:.3f})"
        )
        print(
            f"Dynamic Range: {stats['dynamic_range']['min']:.1f} to {stats['dynamic_range']['max']:.1f} dB (avg: {stats['dynamic_range']['mean']:.1f})"
        )
        print(
            f"Bass Ratio: {stats['bass_ratio']['min']:.2f} to {stats['bass_ratio']['max']:.2f} (avg: {stats['bass_ratio']['mean']:.2f})"
        )

        # Determine optimal mastering targets
        optimal_targets = self._calculate_optimal_targets(stats)
        stats["optimal_targets"] = optimal_targets

        print(f"\n🎯 OPTIMAL MASTERING TARGETS:")
        print(f"  Target LUFS: {optimal_targets['lufs']:.1f}")
        print(f"  Target Peak: {optimal_targets['peak']:.3f}")
        print(f"  Target Bass Ratio: {optimal_targets['bass_ratio']:.2f}")

        return stats

    def _calculate_optimal_targets(self, stats: Dict) -> Dict:
        """Calculate optimal mastering targets based on reference analysis"""
        # Use median values as they're more robust to outliers
        target_lufs = stats["lufs"]["median"]
        target_peak = min(0.95, stats["peak"]["median"])  # Cap at 0.95 for safety
        target_bass_ratio = stats["bass_ratio"]["median"]
        target_high_ratio = stats["high_ratio"]["median"]

        # Ensure targets are within reasonable ranges
        target_lufs = max(-12.0, min(-6.0, target_lufs))  # Club mastering range
        target_bass_ratio = max(0.25, min(0.5, target_bass_ratio))  # Balanced bass
        target_high_ratio = max(0.15, min(0.35, target_high_ratio))  # Adequate highs

        return {
            "lufs": target_lufs,
            "peak": target_peak,
            "bass_ratio": target_bass_ratio,
            "high_ratio": target_high_ratio,
            "description": "Optimized for electronic music based on reference analysis",
        }

    def save_analysis(self, stats: Dict, output_path: str):
        """Save reference analysis to JSON file"""
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"💾 Analysis saved to: {output_path}")


if __name__ == "__main__":
    # Analyze user references
    analyzer = ReferenceAnalyzer()

    references_dir = "references/user_references"
    stats = analyzer.analyze_reference_collection(references_dir)

    if stats:
        output_path = "reference_analysis.json"
        analyzer.save_analysis(stats, output_path)

        print(f"\n🎉 REFERENCE ANALYSIS COMPLETE!")
        print(f"Analyzed {stats['num_tracks']} professional tracks")
        print(f"Results saved to: {output_path}")
    else:
        print("❌ No reference analysis could be performed")
