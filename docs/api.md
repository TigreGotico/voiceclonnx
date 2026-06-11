# API reference

## `vconnx.VoiceCloner`

The public facade. All engines are accessed through this class.

```python
from vconnx import VoiceCloner
```

### Constructor

```python
VoiceCloner(engine: str = "chatterbox", **cfg)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine` | `str` | `"chatterbox"` | Engine alias from the registry |
| `**cfg` | | | Engine-specific keyword arguments forwarded to the adapter constructor |

Raises `KeyError` if `engine` is not in the registry. The error message lists
known engine aliases.

Engine-specific kwargs vary per engine — see the engine guides for the full list.
Common examples: `quantized=True` (knnvc, openvoice), `exaggeration=0.6`
(chatterbox), `k=4` (knnvc), `gl_iters=32` (openvoice).

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
|---|---|
| `audio` | Path to the source WAV file. Any sample rate; 16-bit PCM recommended. |
| `reference_voice` | Path to a short (5–30 s) reference WAV providing the target speaker identity. |
| `out_path` | Destination path for the converted 16-bit WAV. Defaults to `audio` with `_converted` suffix in the same directory. |

Returns the path to the written output file (same as `out_path`).

### Properties

| Property | Type | Description |
|---|---|---|
| `sample_rate` | `int` | Output sample rate in Hz for the active engine |
| `engine` | `str` | Active engine alias |

---

## `vconnx.engines.base.VoiceClonerBase`

Abstract base class for per-engine adapters. Subclass this to add a new engine.

```python
from vconnx.engines.base import VoiceClonerBase
```

### Class attribute

| Attribute | Default | Description |
|---|---|---|
| `_sample_rate` | `16000` | Override in subclass to expose the true output sample rate |

### Constructor

```python
VoiceClonerBase(**cfg)
```

Stores `cfg` in `self._cfg`. Passes unknown kwargs through.

### Abstract method

```python
clone_voice(audio: str, reference_voice: str, out_path: str) -> str
```

Must be implemented by subclasses. Must write a 16-bit WAV to `out_path` and return
the absolute path to the written file.

### Property

```python
sample_rate -> int
```

Returns `self._sample_rate`.

---

## `vconnx.engines.base.EngineEntry`

Frozen-style dataclass describing one registered engine.

```python
from vconnx.engines.base import EngineEntry
```

| Field | Type | Description |
|---|---|---|
| `alias` | `str` | Registry key (e.g. `"chatterbox"`) |
| `adapter_class` | `Type[VoiceClonerBase]` | The adapter class |
| `description` | `str` | Human-readable one-line description |
| `extras` | `str` | pip extras key (e.g. `"chatterbox"` → `pip install vconnx[chatterbox]`) |
| `onnx_native` | `bool` | `True` for all built-in engines |

---

## Registry functions

```python
from vconnx.engines.base import ENGINE_REGISTRY, register_engine, get_engine
```

### `ENGINE_REGISTRY`

`Dict[str, EngineEntry]` — the live registry dict. Populated by importing engine
adapter modules.

### `register_engine`

```python
register_engine(entry: EngineEntry) -> EngineEntry
```

Register an engine entry. Returns `entry` for decorator-style usage at module level.

### `get_engine`

```python
get_engine(alias: str) -> EngineEntry
```

Retrieve a registry entry by alias. Raises `KeyError` with a helpful message listing
known aliases when `alias` is not found.

---

## Adding a custom engine

```python
from vconnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

class MyAdapter(VoiceClonerBase):
    _sample_rate = 16000

    def __init__(self, my_param: float = 1.0, **cfg):
        super().__init__(**cfg)
        self._my_param = my_param
        self._model = None

    def _load(self):
        if self._model is None:
            # lazy-load your ONNX sessions here
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

Then expose `my-engine` in `pyproject.toml` under `[project.optional-dependencies]`
and auto-import the module in `vconnx/__init__.py`.
