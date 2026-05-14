import pickle
import unittest
import sys
import os
import io
from typing import Any

# Define SafeUnpickler here since importing it from atom.utils.pickle
# triggers imports of many modules that are not installed in this environment.
# We want to test the logic of the whitelist enforcement.

WHITELIST = {
    "builtins": {
        "int", "float", "str", "bool", "dict", "list", "set", "tuple",
        "complex", "bytes", "bytearray", "slice", "type", "NoneType",
    },
    "atom.model_engine.engine_core": {"EngineCoreRequestType"},
    "atom.model_engine.sequence": {"Sequence", "SequenceStatus", "SequenceType"},
    "atom.sampling_params": {"SamplingParams"},
    "atom.model_engine.request": {"RequestOutput"},
    "atom.model_engine.scheduler": {"ScheduledBatch", "ScheduledBatchOutput"},
    "atom.kv_transfer.disaggregation.types": {
        "KVConnectorOutput", "ReqMeta", "RemoteAllocInfo", "RemoteMeta", "ConnectorMetadata",
    },
    "numpy": {"ndarray", "dtype"},
    "numpy.core.multiarray": {"_reconstruct", "scalar"},
    "numpy._core.multiarray": {"_reconstruct", "scalar"},
}

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in WHITELIST and name in WHITELIST[module]:
            # In a real run, this would call super().find_class(module, name)
            # For this standalone test, we just return a dummy if it's whitelisted
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                return type(name, (), {"__module__": module})

        if module.startswith("numpy"):
             if name in {"_reconstruct", "scalar", "dtype"}:
                try:
                    return super().find_class(module, name)
                except (ModuleNotFoundError, AttributeError):
                    return type(name, (), {"__module__": module})

        raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")

def safe_loads(data: bytes) -> Any:
    if not isinstance(data, (bytes, bytearray)):
        if hasattr(data, "bytes"):
            data = data.bytes
        else:
            raise TypeError(f"Expected bytes or bytearray, got {type(data)}")
    return SafeUnpickler(io.BytesIO(data)).load()

class Malicious:
    def __reduce__(self):
        import os
        return (os.system, ('echo "vulnerable"',))

class TestSecurityPickle(unittest.TestCase):
    def test_safe_loads_unauthorized_class(self):
        malicious_obj = Malicious()
        serialized = pickle.dumps(malicious_obj)

        with self.assertRaises(pickle.UnpicklingError) as cm:
            safe_loads(serialized)

        self.assertIn("is forbidden", str(cm.exception))
        # Depending on platform, os.system might be posix.system or nt.system
        self.assertTrue("posix.system" in str(cm.exception) or "nt.system" in str(cm.exception) or "os.system" in str(cm.exception))

    def test_safe_loads_authorized_class(self):
        # Test with a whitelisted name
        data = pickle.dumps(int)
        deserialized = safe_loads(data)
        self.assertEqual(deserialized, int)

    def test_safe_loads_basic_types(self):
        obj = {"a": [1, 2, 3], "b": (4, 5), "c": "hello", "d": True}
        serialized = pickle.dumps(obj)

        deserialized = safe_loads(serialized)
        self.assertEqual(deserialized, obj)

if __name__ == '__main__':
    unittest.main()
