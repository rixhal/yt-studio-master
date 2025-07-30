#!/usr/bin/env python3
"""
🎯 REFERENCE-OPTIMIZED MASTERING PRESETS
Dynamic presets optimized based on user's professional reference collection

Features:
- Reference-based target optimization (-9.6 LUFS for electronic music)
- Adaptive bass/high frequency balance (88% bass, 1.7% highs)
- Professional peak limiting (0.95 target)
- Electronic music specialized processing
"""

import json
import os
from typing import Any, Dict


class ReferenceOptimizedPresets:
    """Dynamic mastering presets optimized from reference analysis"""

    def __init__(self, reference_analysis_path: str = "reference_analysis.json"):
        self.analysis = self._load_reference_analysis(reference_analysis_path)

    def _load_reference_analysis(self, path: str) -> Dict:
        """Load reference analysis data"""
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        else:
            print(f"⚠️ Reference analysis not found: {path}")
            return self._get_default_analysis()

    def _get_default_analysis(self) -> Dict:
        """Fallback analysis if reference data not available"""
        return {
            "optimal_targets": {
                "lufs": -8.0,
                "peak": 0.95,
                "bass_ratio": 0.4,
                "high_ratio": 0.25,
                "description": "Default electronic music targets",
            },
            "lufs": {"median": -8.0},
            "bass_ratio": {"median": 0.4},
            "high_ratio": {"median": 0.25},
        }

    def get_reference_optimized_preset(
        self, style: str = "electronic"
    ) -> Dict[str, Any]:
        """Get mastering preset optimized for your reference collection"""
        targets = self.analysis.get("optimal_targets", {})

        # Reference-based targets from your collection
        target_lufs = targets.get("lufs", -9.6)
        target_peak = targets.get("peak", 0.95)
        bass_ratio = self.analysis.get("bass_ratio", {}).get("median", 0.88)
        high_ratio = self.analysis.get("high_ratio", {}).get("median", 0.017)

        preset = {
            "name": f"Reference_Optimized_{style.title()}",
            "description": f'Optimized from {self.analysis.get("num_tracks", 0)} professional references',
            # Core targets based on YOUR references
            "target_lufs": target_lufs,  # -9.6 LUFS (your references median)
            "target_peak": target_peak,  # 0.95 peak
            "target_rms": 0.33,  # Derived from your reference RMS values
            # Frequency balance from your references
            "frequency_targets": {
                "bass_ratio": min(0.5, bass_ratio),  # Cap at 0.5 for safety
                "mid_ratio": 1.0 - min(0.5, bass_ratio) - min(0.3, high_ratio * 10),
                "high_ratio": min(0.3, high_ratio * 10),  # Scale up high ratio
                "bass_boost_db": 2.0,  # Moderate bass boost
                "high_boost_db": 1.5,  # Gentle high boost
            },
            # Dynamics based on reference analysis
            "dynamics": {
                "compression_ratio": 3.0,  # Moderate compression
                "compression_threshold": -12.0,
                "compression_attack": 3.0,
                "compression_release": 100.0,
                "target_dynamic_range": 10.3,  # Your references average
            },
            # EQ optimizations
            "eq_bands": [
                {"freq": 60, "gain": 1.5, "q": 0.7},  # Sub bass warmth
                {"freq": 120, "gain": 1.0, "q": 0.5},  # Bass punch
                {"freq": 300, "gain": -0.5, "q": 0.8},  # Mud reduction
                {"freq": 1000, "gain": 0.5, "q": 0.5},  # Presence
                {"freq": 3000, "gain": 1.0, "q": 0.7},  # Clarity
                {"freq": 8000, "gain": 1.5, "q": 0.5},  # Air
                {"freq": 12000, "gain": 1.0, "q": 0.7},  # Sparkle
            ],
            # Stereo processing
            "stereo": {
                "stereo_width": 1.1,  # Slight widening
                "bass_mono_freq": 120,  # Mono bass below 120Hz
                "stereo_enhancement": True,
            },
            # Saturation and warmth
            "saturation": {
                "drive": 2.0,
                "type": "tape",
                "mix": 15.0,  # 15% saturation mix
            },
            # Limiting (final stage)
            "limiter": {
                "threshold": target_peak,
                "release": 30.0,
                "isr": 4,  # Internal sample rate multiplier
                "lookahead": 5.0,
            },
            # Processing chain order
            "chain_order": [
                "eq",
                "compression",
                "saturation",
                "stereo_enhancement",
                "limiter",
            ],
            # Specialized for electronic music
            "genre_specific": {
                "sub_bass_enhancement": True,
                "transient_preservation": True,
                "electronic_character": True,
                "club_ready": True,
            },
        }

        return preset

    def get_style_variations(self) -> Dict[str, Dict]:
        """Get multiple preset variations based on your references"""
        base_preset = self.get_reference_optimized_preset()

        variations = {
            "Club_Loud": {
                **base_preset,
                "name": "Reference_Club_Loud",
                "target_lufs": -7.8,  # Loudest from your references
                "description": "Club-ready loud master based on GTI - Dumtek style",
                "dynamics": {
                    **base_preset["dynamics"],
                    "compression_ratio": 4.0,
                    "compression_threshold": -8.0,
                },
            },
            "Dynamic_Master": {
                **base_preset,
                "name": "Reference_Dynamic",
                "target_lufs": -12.5,  # Most dynamic from your references
                "description": "Dynamic master based on Seta Loto - 5MeODMT style",
                "dynamics": {
                    **base_preset["dynamics"],
                    "compression_ratio": 2.0,
                    "compression_threshold": -18.0,
                    "target_dynamic_range": 12.2,
                },
            },
            "Balanced_Electronic": {
                **base_preset,
                "name": "Reference_Balanced",
                "target_lufs": -9.6,  # Your references median
                "description": "Balanced electronic master optimized from all references",
            },
            "Experimental": {
                **base_preset,
                "name": "Reference_Experimental",
                "target_lufs": -10.5,  # Olsvangèr style
                "description": "Experimental/ambient style based on Olsvangèr references",
                "frequency_targets": {
                    **base_preset["frequency_targets"],
                    "bass_ratio": 0.45,  # Less bass for experimental
                    "high_ratio": 0.25,  # More highs
                },
                "stereo": {
                    **base_preset["stereo"],
                    "stereo_width": 1.3,  # Wider stereo field
                },
            },
        }

        return variations

    def print_reference_summary(self):
        """Print summary of reference analysis"""
        if "num_tracks" in self.analysis:
            print(f"🎯 REFERENCE-OPTIMIZED MASTERING")
            print(f"Based on {self.analysis['num_tracks']} professional tracks")
            print(
                f"LUFS Range: {self.analysis['lufs']['min']:.1f} to {self.analysis['lufs']['max']:.1f}"
            )
            print(
                f"Optimal Target: {self.analysis['optimal_targets']['lufs']:.1f} LUFS"
            )
            print(f"Peak Target: {self.analysis['optimal_targets']['peak']:.3f}")
            print(f"Bass Character: {self.analysis['bass_ratio']['median']:.2f} ratio")
        else:
            print("⚠️ No reference analysis available, using defaults")


if __name__ == "__main__":
    # Test the reference-optimized presets
    presets = ReferenceOptimizedPresets()
    presets.print_reference_summary()

    print("\n🎛️ AVAILABLE PRESETS:")
    variations = presets.get_style_variations()

    for name, preset in variations.items():
        print(f"\n{name}:")
        print(f"  Target LUFS: {preset['target_lufs']:.1f}")
        print(f"  Description: {preset['description']}")
