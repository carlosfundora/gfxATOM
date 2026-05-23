import time
import numpy as np

# A simple mock to verify the latency overhead of get_inputs() vs cached

class MockNodeArg:
    def __init__(self, name):
        self.name = name

class MockInferenceSession:
    def get_inputs(self):
        # Simulate ONNX C++ boundary crossing overhead
        return [MockNodeArg("input_ids"), MockNodeArg("position_ids"), MockNodeArg("exaggeration")]

    def run(self, output_names, input_feed):
        # Return dummy embedding
        return [np.zeros((1, 1, 64), dtype=np.float32)]

class ChatterboxServiceBefore:
    def __init__(self):
        self._embed_tokens = MockInferenceSession()

    def embed_single_token(self, token_id, exaggeration=0.5):
        embed_input_names = [i.name for i in self._embed_tokens.get_inputs()]
        ort_inputs = {"input_ids": token_id}
        if "position_ids" in embed_input_names:
            ort_inputs["position_ids"] = np.zeros_like(token_id)
        if "exaggeration" in embed_input_names:
            ort_inputs["exaggeration"] = np.array([exaggeration], dtype=np.float32)
        return self._embed_tokens.run(None, ort_inputs)[0]

class ChatterboxServiceAfter:
    def __init__(self):
        self._embed_tokens = MockInferenceSession()
        self._embed_input_names = [i.name for i in self._embed_tokens.get_inputs()]

    def embed_single_token(self, token_id, exaggeration=0.5):
        ort_inputs = {"input_ids": token_id}
        if "position_ids" in self._embed_input_names:
            ort_inputs["position_ids"] = np.zeros_like(token_id)
        if "exaggeration" in self._embed_input_names:
            ort_inputs["exaggeration"] = np.array([exaggeration], dtype=np.float32)
        return self._embed_tokens.run(None, ort_inputs)[0]

def main():
    token_id = np.array([[1000]], dtype=np.int64)
    iterations = 10000

    before = ChatterboxServiceBefore()
    t0 = time.time()
    for _ in range(iterations):
        before.embed_single_token(token_id)
    t_before = time.time() - t0

    after = ChatterboxServiceAfter()
    t1 = time.time()
    for _ in range(iterations):
        after.embed_single_token(token_id)
    t_after = time.time() - t1

    print(f"Benchmark: TTS Latency Overhead (10000 iterations)")
    print(f"Before (uncached get_inputs): {t_before*1000:.2f} ms")
    print(f"After (cached input_names): {t_after*1000:.2f} ms")
    print(f"Time saved per step: {(t_before - t_after)/iterations*1000000:.2f} µs")

if __name__ == "__main__":
    main()
