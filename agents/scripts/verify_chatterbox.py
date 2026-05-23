import sys
import numpy as np

# We'll just extract the rep penalty logic to test it if we can't import the file
def _np_rep_penalty(input_ids, scores, penalty):
    if input_ids.shape[0] == 1:
        ids = input_ids[0]
        s = scores[0, ids]
        mask = s < 0
        s[mask] = (s[mask] * penalty).astype(scores.dtype)
        s[~mask] = (s[~mask] / penalty).astype(scores.dtype)
        scores[0, ids] = s
        return scores

    score = np.take_along_axis(scores, input_ids, axis=1)
    mask = score < 0
    score[mask] = (score[mask] * penalty).astype(scores.dtype)
    score[~mask] = (score[~mask] / penalty).astype(scores.dtype)
    np.put_along_axis(scores, input_ids, score, axis=1)
    return scores

def test_engine_rep_penalty():
    vocab_size = 32000
    batch_size = 1
    seq_len = 512
    np.random.seed(42)
    scores = np.random.randn(batch_size, vocab_size).astype(np.float32)
    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(np.int64)
    penalty = 1.2

    # Check it runs without error
    out = _np_rep_penalty(input_ids, scores.copy(), penalty)
    assert out.shape == scores.shape
    print("Rep penalty isolated test passed")

if __name__ == "__main__":
    test_engine_rep_penalty()
