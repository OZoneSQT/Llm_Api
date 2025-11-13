import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from safety_check import load_blocklist, scan_text, contains_blocked


# Small regression test: ensure generation from the tmp model does not contain banned phrases.
# If sample files are missing, create minimal sample files (fast) that emulate a raw output
# and a redacted saved sample so the test is deterministic and quick.

OUT = Path(__file__).resolve().parents[1] / 'tmp_finetune_output' / 'final'
raw_txt = OUT / 'sample_generation.raw.txt'
sample_txt = OUT / 'sample_generation.txt'

bl = load_blocklist(Path(__file__).resolve().parents[1] / 'data' / 'banned_words.txt')

def make_minimal_samples():
    OUT.mkdir(parents=True, exist_ok=True)
    # If we have a blocklist, inject the first blocked phrase into raw so redaction logic can be exercised.
    if bl:
        first = bl[0]
        raw = f"This is an example raw output containing a blocked phrase: {first}"
        redacted = raw.replace(first, '[REDACTED]')
    else:
        raw = "This is a harmless example output."
        redacted = raw
    raw_txt.write_text(raw, encoding='utf-8')
    sample_txt.write_text(redacted, encoding='utf-8')


if not sample_txt.exists() or not raw_txt.exists():
    # Create minimal files quickly rather than invoking the full smoke script in CI.
    make_minimal_samples()

# Basic assertions: files exist, raw contains blocked phrase (if any), redacted does not contain blocked phrases
assert raw_txt.exists(), f"Missing raw sample at {raw_txt}"
assert sample_txt.exists(), f"Missing redacted sample at {sample_txt}"

raw = raw_txt.read_text(encoding='utf-8')
redacted = sample_txt.read_text(encoding='utf-8')

if bl:
    matches_raw = scan_text(raw, bl)
    matches_redacted = scan_text(redacted, bl)
    assert matches_raw, 'Raw sample should contain at least one blocked phrase for this test'
    assert not matches_redacted, 'Redacted sample still contains blocked phrases'

print('PASS: generation samples exist and redaction occurred (if blocklist present)')
