import importlib, sys, os
importlib.invalidate_caches()
try:
    import datasets
    print('datasets.__file__ ->', getattr(datasets, '__file__', None))
    print('datasets.__package__ ->', getattr(datasets, '__package__', None))
    print('datasets.__path__ ->', getattr(datasets, '__path__', None))
    print('type(datasets) ->', type(datasets))
    print('repr(datasets) ->', repr(datasets)[:200])
except Exception as e:
    print('import datasets failed:', repr(e))
print('cwd ->', os.getcwd())
print('sys.path[0..6] ->')
for i, p in enumerate(sys.path[:7]):
    print(i, p)
