import io
import pickle
from typing import Any

# Allowed modules and classes for unpickling
# This whitelist covers the core data structures used in ATOM's inter-process communication.
WHITELIST = {
    "builtins": {
        "int",
        "float",
        "str",
        "bool",
        "dict",
        "list",
        "set",
        "tuple",
        "complex",
        "bytes",
        "bytearray",
        "slice",
        "type",
        "NoneType",
    },
    "atom.model_engine.engine_core": {"EngineCoreRequestType"},
    "atom.model_engine.sequence": {"Sequence", "SequenceStatus", "SequenceType"},
    "atom.sampling_params": {"SamplingParams"},
    "atom.model_engine.request": {"RequestOutput"},
    "atom.model_engine.scheduler": {"ScheduledBatch", "ScheduledBatchOutput"},
    "atom.kv_transfer.disaggregation.types": {
        "KVConnectorOutput",
        "ReqMeta",
        "RemoteAllocInfo",
        "RemoteMeta",
        "ConnectorMetadata",
    },
    "numpy": {
        "ndarray",
        "dtype",
    },
    "numpy.core.multiarray": {
        "_reconstruct",
        "scalar",
    },
    "numpy._core.multiarray": {
        "_reconstruct",
        "scalar",
    },
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in WHITELIST and name in WHITELIST[module]:
            return super().find_class(module, name)

        # Allow some common numpy submodules that might be used during reconstruction
        if module.startswith("numpy"):
            if name in {"_reconstruct", "scalar", "dtype"}:
                return super().find_class(module, name)

        raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")


def safe_loads(data: bytes) -> Any:
    """
    Safely deserialize an object from bytes using a restricted Unpickler.
    """
    if not isinstance(data, (bytes, bytearray)):
        # Handle zmq.Frame if passed directly
        if hasattr(data, "bytes"):
            data = data.bytes
        else:
            raise TypeError(f"Expected bytes or bytearray, got {type(data)}")

    return SafeUnpickler(io.BytesIO(data)).load()
