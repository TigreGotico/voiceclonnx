# Examples

| File | Description | Requirements |
|---|---|---|
| [`basic_clone.py`](basic_clone.py) | Facade demo with knnvc: synthesise two TTS clips via edge-tts and run voice conversion | `vconnx`, `edge-tts` |
| [`cli_batch.sh`](cli_batch.sh) | Bash loop — convert a folder of WAVs using the `vconnx clone` CLI | any engine installed |
| [`quantized_low_memory.py`](quantized_low_memory.py) | INT8 quantized variants for knnvc and openvoice: timing + memory comparison | `vconnx` |
| [`local_only_engine.md`](local_only_engine.md) | Walkthrough: run the conversion script yourself and configure `model_dir` for non-redistributable weights | `vconnx[convert]` |

## Quick start

```bash
# knnvc basic demo
pip install vconnx edge-tts
python examples/basic_clone.py

# Batch convert a folder
chmod +x examples/cli_batch.sh
./examples/cli_batch.sh ./my_wavs reference.wav ./out chatterbox

# Quantized comparison
pip install vconnx
python examples/quantized_low_memory.py source.wav reference.wav
```
