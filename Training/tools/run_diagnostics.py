import sys
import torch
from transformers import AutoModelForCausalLM
from Training.tools.adapter_utils import load_tokenizer, load_model

MODEL = "dphn/Dolphin3.0-Llama3.2-1B"

print('Using Python executable:', sys.executable)
print('Torch version:', torch.__version__)

try:
    print('\nLoading tokenizer (trust_remote_code=True) ...')
    tok = load_tokenizer(MODEL, local_files_only=False, trust_remote_code=True)
    print('Tokenizer class:', tok.__class__)
    print('vocab_size:', getattr(tok, 'vocab_size', None))
    print('pad_token_id:', tok.pad_token_id, 'eos_token_id:', tok.eos_token_id)
except Exception as e:
    print('Tokenizer load failed:', repr(e))
    raise

try:
    print('\nLoading model (trust_remote_code=True) ...')
    model = load_model(MODEL, loader=AutoModelForCausalLM.from_pretrained, local_files_only=False, trust_remote_code=True)
    print('Model loaded. config.vocab_size:', model.config.vocab_size)
except Exception as e:
    print('Model load failed:', repr(e))
    raise

# Round-trip tokenization test
text = "Hello! This is a short tokenization test."
print('\nRound-trip test input:', text)
ids = tok(text).input_ids
print('Encoded ids (first 40):', ids[:40])
dec = tok.decode(ids, skip_special_tokens=True)
print('Decoded:', dec)

# Small generation test (use CPU if CUDA not available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('\nUsing device for generation:', device)
model.to(device)

prompt = "Write a short friendly instruction about gardening:"
inputs = tok(prompt, return_tensors='pt').to(device)
print('\nGenerating a short sample...')
try:
    out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.8, top_p=0.9, top_k=40, repetition_penalty=1.1, no_repeat_ngram_size=3, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    txt = tok.decode(out[0], skip_special_tokens=True)
    print('\nGeneration result:')
    print(txt)
except Exception as e:
    print('Generation failed:', repr(e))
    raise

print('\nDiagnostics complete.')
