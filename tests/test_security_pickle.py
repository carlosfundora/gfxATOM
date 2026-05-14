import pickle
import unittest
import sys
import os
import io

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules that might be missing or trigger heavy imports
from unittest.mock import MagicMock
sys.modules['torch'] = MagicMock()
sys.modules['zmq'] = MagicMock()
sys.modules['zmq.asyncio'] = MagicMock()
sys.modules['aiter'] = MagicMock()
sys.modules['aiter.dist.shm_broadcast'] = MagicMock()
sys.modules['aiter.dist.parallel_state'] = MagicMock()
sys.modules['aiter.dist.utils'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['transformers'] = MagicMock()

# Now we can import safe_loads
from atom.utils.pickle import safe_loads

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
        self.assertIn("os.system", str(cm.exception))

    def test_safe_loads_basic_types(self):
        obj = {"a": [1, 2, 3], "b": (4, 5), "c": "hello", "d": True}
        serialized = pickle.dumps(obj)

        deserialized = safe_loads(serialized)
        self.assertEqual(deserialized, obj)

if __name__ == '__main__':
    unittest.main()
