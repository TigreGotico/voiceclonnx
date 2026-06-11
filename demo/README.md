# Engine comparison demos

Same source speech, same reference voices, every engine — listen and compare
without running any code (GitHub renders the players inline).

## Inputs

| file | what it is |
|---|---|
| [`source.wav`](source.wav) | the speech to convert (edge-tts `en-US-GuyNeural`) |
| [`reference_aria.wav`](reference_aria.wav) | reference voice A (edge-tts `en-US-AriaNeural`) |
| [`reference_sonia.wav`](reference_sonia.wav) | reference voice B (edge-tts `en-GB-SoniaNeural`) |

## Converted outputs

One file per engine × reference: `outputs/<engine>__<reference>.wav` —
the source utterance spoken in the reference voice, as that engine hears it.

| engine | → aria | → sonia |
|---|---|---|
| chatterbox | [`outputs/chatterbox__aria.wav`](outputs/chatterbox__aria.wav) | [`outputs/chatterbox__sonia.wav`](outputs/chatterbox__sonia.wav) |
| knnvc | [`outputs/knnvc__aria.wav`](outputs/knnvc__aria.wav) | [`outputs/knnvc__sonia.wav`](outputs/knnvc__sonia.wav) |
| openvoice | [`outputs/openvoice__aria.wav`](outputs/openvoice__aria.wav) | [`outputs/openvoice__sonia.wav`](outputs/openvoice__sonia.wav) |
| triaan | [`outputs/triaan__aria.wav`](outputs/triaan__aria.wav) | [`outputs/triaan__sonia.wav`](outputs/triaan__sonia.wav) |

Any-to-ONE engines convert to a fixed voice model instead of a reference wav
(one column per demo model):

| engine | output | target model |
|---|---|---|
| rvc | [`outputs/rvc__woman1.wav`](outputs/rvc__woman1.wav) | `ozada/onnx_rvc::woman_1.onnx` (community, MIT) |

Rows are added as engines land. Engines with `local-only-weights` upstreams
appear here too — the demo outputs are generated locally by maintainers and
committing a converted wav redistributes no model weights.

## Regenerating

```bash
pip install "vconnx[knnvc,chatterbox]" edge-tts
python demo/generate_demos.py            # all registered engines
python demo/generate_demos.py --engines knnvc
```

The script synthesizes the inputs deterministically (fixed texts/voices) and
overwrites `outputs/`; inputs are kept if already present so every engine is
judged on identical audio.
