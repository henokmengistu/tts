import subprocess
import sys


def test_cli_normalize():
    out = subprocess.run([sys.executable, "-m", "amharic_tts.cli", "normalize", "ዶ/ር አበበ 25% አለ።"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "ዶክተር አበበ ሃያ አምስት በመቶ አለ."


def test_cli_voices_list():
    out = subprocess.run([sys.executable, "-m", "amharic_tts.cli", "voices", "list"],
                         capture_output=True, text=True, check=True).stdout
    assert "selam" in out and "dawit" in out
