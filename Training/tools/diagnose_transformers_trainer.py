import traceback
try:
    import importlib
    mod = importlib.import_module('transformers.trainer')
    print('transformers.trainer ->', getattr(mod, '__file__', None))
    print('Names:', [n for n in dir(mod) if not n.startswith('_')][:200])
    try:
        from transformers.trainer import Trainer
        print('Trainer imported from transformers.trainer')
    except Exception as e:
        print('trainer.Trainer import failed:', repr(e))
        traceback.print_exc()
except Exception as e:
    print('Failed to import transformers.trainer:', repr(e))
    traceback.print_exc()
