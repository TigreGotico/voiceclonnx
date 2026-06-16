# API reference

## `voiceclonnx.VoiceCloner`

The public facade. All engines are accessed through this class.

```python
from voiceclonnx import VoiceCloner
```

### Constructor

```python
VoiceCloner(engine: str = "chatterbox", **cfg)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | `str` | `"chatterbox"` | Engine alias from the registry |
| `**cfg` | | | Engine-specific keyword arguments forwarded to the adapter constructor |

Raises `KeyError` if `engine` is not in the registry; the error message lists
all known aliases.

Engine-specific kwargs vary per engine — see [engines/](engines/) for the full
list per engine. Common examples:

| kwarg | Engines | Description |
|-------|---------|-------------|
| `quantized=True` | all except `chatterbox` | Load INT8 `*_q8.onnx` variants |
| `exaggeration=0.5` | `chatterbox` | Voice conditioning strength |
| `k=4` | `knnvc`, `focalcodec` | Number of nearest neighbours |
| `gl_iters=32` | `openvoice` | Griffin-Lim iterations |
| `ode_steps=10` | `cosyvoice` | ODE Euler steps for flow decoder |
| `default_model=...` | `rvc` | Default voice model path |

### Methods

#### `clone_voice`

```python
clone_voice(
    audio: str,
    reference_voice: str,
    out_path: Optional[str] = None,
) -> str
```

Convert the voice in `audio` to sound like `reference_voice`.

| Parameter | Description |
|-----------|-------------|
| `audio` | Path to the source WAV file. Any sample rate; 16-bit PCM recommended. |
| `reference_voice` | Path to a short (5–30 s) reference WAV providing the target speaker identity. For `rvc`: path to an `.onnx` voice model or HF repo ID — **not** an audio file. |
| `out_path` | Destination path for the converted 16-bit WAV. Defaults to `audio` with `_converted` suffix in the same directory. |

Returns the path to the written output file.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `sample_rate` | `int` | Output sample rate in Hz for the active engine |
| `engine` | `str` | Active engine alias |

---

## `voiceclonnx.engines.base.VoiceClonerBase`

Abstract base class for per-engine adapters. Subclass this to add a new engine.

```python
from voiceclonnx.engines.base import VoiceClonerBase
```

### Class attribute

| Attribute | Default | Description |
|-----------|---------|-------------|
| `_sample_rate` | `16000` | Override in subclass to expose the true output sample rate |

### Constructor

```python
VoiceClonerBase(**cfg)
```

Stores `cfg` in `self._cfg`. Unknown kwargs are passed through.

### Abstract method

```python
clone_voice(audio: str, reference_voice: str, out_path: str) -> str
```

Must be implemented by subclasses. Must write a 16-bit WAV to `out_path` and
return the absolute path to the written file.

### Property

```python
sample_rate -> int
```

Returns `self._sample_rate`.

---

## `voiceclonnx.engines.base.EngineEntry`

Frozen-style dataclass describing one registered engine.

```python
from voiceclonnx.engines.base import EngineEntry
```

| Field | Type | Description |
|-------|------|-------------|
| `alias` | `str` | Registry key (e.g. `"facodec"`) |
| `adapter_class` | `Type[VoiceClonerBase]` | The adapter class |
| `description` | `str` | Human-readable one-line description |
| `extras` | `str` | pip extras key |
| `onnx_native` | `bool` | `True` for all built-in engines |

---

## Registry functions

```python
from voiceclonnx.engines.base import ENGINE_REGISTRY, register_engine, get_engine
```

### `ENGINE_REGISTRY`

`Dict[str, EngineEntry]` — the live registry dict. Populated by importing engine
adapter modules (done automatically in `voiceclonnx/__init__.py` for all 9
built-in engines).

### `register_engine`

```python
register_engine(entry: EngineEntry) -> EngineEntry
```

Register an engine entry. Returns `entry` for decorator-style usage at module level.

### `get_engine`

```python
get_engine(alias: str) -> EngineEntry
```

Retrieve a registry entry by alias. Raises `KeyError` with a helpful message
listing known aliases when `alias` is not found.

---

## Adding a custom engine

```python
from voiceclonnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

class MyAdapter(VoiceClonerBase):
    _sample_rate = 16000

    def __init__(self, my_param: float = 1.0, **cfg):
        super().__init__(**cfg)
        self._my_param = my_param
        self._model = None

    def _load(self):
        if self._model is None:
            # lazy-load ONNX sessions here
            pass

    def clone_voice(self, audio: str, reference_voice: str, out_path: str) -> str:
        self._load()
        # ... run inference, write out_path ...
        return out_path

register_engine(EngineEntry(
    alias="my-engine",
    adapter_class=MyAdapter,
    description="My custom engine",
    extras="my-engine",
    onnx_native=True,
))
```

Then add the auto-import to `voiceclonnx/__init__.py`. For a full walkthrough
of the export → parity → quantize → push → adapter pipeline, see
[converting.md](converting.md).
