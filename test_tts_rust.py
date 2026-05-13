import sys
import os
import time
import subprocess
import soundfile as sf
import numpy as np
from pathlib import Path

# Add project root to sys.path to allow importing atom
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from atom.audio.chatterbox.engine import ChatterboxEngine
from atom.audio.text_splitter import SentenceSplitter
from atom.audio.utils import audio_to_bytes

def main():
    print("Testing SentenceSplitter (Rust)...")
    splitter = SentenceSplitter(min_sentence_length=2)
    sentences = splitter.add_text("Hello world! This is a test. We will generate speech. ")
    sentences += splitter.add_text("And another sentence here. ")
    sentences.append(splitter.flush())
    print("Split sentences:", sentences)
    
    # Path to Chatterbox models
    model_dir = "/home/local/Projects/models/huggingface/models--onnx-community--chatterbox-ONNX/snapshots/main"
    backbone_dir = "/home/local/Projects/models/huggingface/models--onnx-community--chatterbox-backbone"
    
    # Fallback to local paths if needed
    if not os.path.exists(model_dir):
        # We can just try running it with None and hope it throws a useful error or we construct paths
        print(f"Model dir {model_dir} not found. Skipping TTS generation.")
        return

    print("Initializing ChatterboxEngine...")
    engine = ChatterboxEngine(
        model_dir=model_dir,
        backbone_dir=backbone_dir if os.path.exists(backbone_dir) else None,
        device="cuda:0",
        dtype="float16"
    )
    engine.load()
    
    test_text = (
        "The stars are incredibly bright tonight. I've always loved looking at the night sky. "
        "It makes you realize just how vast the universe really is! Sometimes, I wonder if anyone is looking back at us. "
        "Either way, it's a beautiful sight to behold."
    )
    
    print(f"Generating audio for text: '{test_text}'")
    wav, metrics = engine.generate(
        text=test_text,
        max_tokens=1024,
        temperature=0.8,
    )
    
    print(f"Generation complete! Metrics: {metrics}")
    
    out_path = "test_tts_rust.wav"
    sf.write(out_path, wav, 24000)
    print(f"Saved audio to {out_path}")
    
    print("Playing audio over speakers...")
    try:
        subprocess.run(["aplay", out_path], check=True)
    except Exception as e:
        print(f"Failed to play audio using aplay: {e}")
        print("Please play the file manually.")

if __name__ == "__main__":
    main()
