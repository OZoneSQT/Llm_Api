from pathlib import Path
p=Path('Training/tmp_finetune_output/final/sample_generation.txt')
print('path', p.resolve())
print('exists', p.exists())
if p.exists():
    print('size', p.stat().st_size)
    print('----- start sample -----')
    print(p.read_text(encoding='utf-8', errors='replace')[:1000])
    print('----- end sample -----')
else:
    print('final dir listing:')
    d=Path('Training/tmp_finetune_output/final')
    for x in sorted(d.iterdir()):
        print(' -', x.name)
