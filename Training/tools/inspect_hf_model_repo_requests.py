import sys
import requests
from pathlib import Path

repo = "UnfilteredAI/NSFW-3B"
api_url = f"https://huggingface.co/api/models/{repo}"
raw_base = f"https://huggingface.co/{repo}/resolve/main"

print('Querying HF API for', repo)
resp = requests.get(api_url, timeout=30)
if resp.status_code != 200:
    print('Failed to query HF API:', resp.status_code, resp.text)
    sys.exit(1)

info = resp.json()
files = []
if 'siblings' in info:
    files = [s['rfilename'] for s in info['siblings']]
elif 'files' in info:
    files = [f['name'] for f in info['files']]

print(f"Found {len(files)} files in repo")
for f in files:
    print(f)

interesting = ['config.json', 'README.md', 'README', 'requirements.txt']
# also any modeling_*.py or modeling.py
for f in files:
    if f.startswith('modeling') and f.endswith('.py'):
        interesting.append(f)
    if f.endswith('.py') and 'model' in f and f not in interesting:
        # include potential custom modeling files
        interesting.append(f)

for name in interesting:
    if name in files:
        url = f"{raw_base}/{name}"
        print(f"\n--- {name} ({url}) ---\n")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            text = r.text
            lines = text.splitlines()
            if len(lines) > 400:
                print('\n'.join(lines[:400]))
                print('\n... (truncated) ...\n')
            else:
                print(text)
        else:
            print(f"Failed to download {name}: {r.status_code}")
    else:
        print(f"\n--- {name} not present in repo ---\n")

print('\nDone')
