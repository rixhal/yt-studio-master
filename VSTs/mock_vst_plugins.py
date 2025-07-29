# Mock VST Plugin for Testing
# This simulates Neutron 5 behavior for development/testing

import time
import json
from pathlib import Path


class MockNeutron5:
    """Mock Neutron 5 plugin for testing without real VST"""

    def __init__(self):
        self.name = "MockNeutron5"
        self.version = "Test"

    def process(self, input_file, output_file, preset_data=None):
        """Simulate Neutron 5 processing"""
        print(f"🔬 Mock Neutron 5 processing: {Path(input_file).name}")

        if preset_data:
            print(f"   Using preset with {len(preset_data.get('modules', {}))} modules")

        # Simulate processing time
        time.sleep(2)

        # Copy input to output (no real processing)
        import shutil

        shutil.copy2(input_file, output_file)

        print("✅ Mock Neutron 5 processing completed")
        return True


class MockOzone11:
    """Mock Ozone 11 plugin for testing without real VST"""

    def __init__(self):
        self.name = "MockOzone11"
        self.version = "Test"

    def process(self, input_file, output_file, preset_data=None):
        """Simulate Ozone 11 processing"""
        print(f"🎚️ Mock Ozone 11 mastering: {Path(input_file).name}")

        if preset_data:
            modules = preset_data.get("modules", {})
            enabled_modules = [
                m for m, settings in modules.items() if settings.get("enabled", False)
            ]
            print(f"   Using modules: {', '.join(enabled_modules)}")

        # Simulate processing time
        time.sleep(3)

        # Copy input to output (no real processing)
        import shutil

        shutil.copy2(input_file, output_file)

        print("✅ Mock Ozone 11 mastering completed")
        return True
