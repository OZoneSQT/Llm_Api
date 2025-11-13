import traceback
try:
    import transformers
    print('transformers.__file__ ->', getattr(transformers, '__file__', None))
    print('transformers.__version__ ->', getattr(transformers, '__version__', None))
    print('Available names in transformers package:', ', '.join([n for n in dir(transformers) if not n.startswith('_')])[:1000])
    try:
        from transformers import Trainer
        print('Trainer import: OK')
    except Exception as e:
        print('Trainer import failed:', repr(e))
        traceback.print_exc()
except Exception as e:
    print('Failed to import transformers:', repr(e))
    traceback.print_exc()
