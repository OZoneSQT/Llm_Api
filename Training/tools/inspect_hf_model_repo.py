import sys
from huggingface_hub import hf_hub_list, hf_hub_download
from pathlib import Path

def print_file(name, repo_id):
    try:
        path = hf_hub_download(repo_id, name)
        text = Path(path).read_text(encoding='utf-8', errors='replace')
        print(f"\n--- {name} ---\n")
        # limit output length
        lines = text.splitlines()
        if len(lines) > 400:
            print('\n'.join(lines[:400]))
            print('\n... (truncated) ...\n')
        else:
            print(text)
    except Exception as e:
        print(f"Could not download {name}: {e}")

repo = "UnfilteredAI/NSFW-3B"
print("Repo ID:", repo)
try:
    files = hf_hub_list(repo)
    print(f"Found {len(files)} files in repo")
    for f in files:
        print(f.name)
except Exception as e:
    print('Failed to list files:', e)
    sys.exit(1)

# Files of interest
interesting = ['config.json', 'README.md', 'README', 'requirements.txt', 'modeling_stablelm.py', 'modeling.py']
# Also include any file that looks like modeling_*.py
for f in files:
    if f.name.startswith('modeling') and f.name.endswith('.py') and f.name not in interesting:
        interesting.append(f.name)

for name in interesting:
    if any(f.name == name for f in files):
        print_file(name, repo)
    else:
        print(f"\n--- {name} not present in repo ---\n")

print('\nInspection complete')
