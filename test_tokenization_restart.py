#!/usr/bin/env python3
"""
Test script to demonstrate the tokenization restart functionality.

This script shows how to use the resume feature in ethos_tokenize.
"""

import subprocess
import sys
from pathlib import Path

def test_tokenization_restart():
    """Test the tokenization restart functionality."""
    
    # Example usage of ethos_tokenize with resume functionality
    print("=== ETHOS Tokenization Restart Test ===\n")
    
    print("1. First run (without resume):")
    print("ethos_tokenize -m worker='range(0,7)' \\")
    print("    input_dir=data/mimic-2.2-meds/data/train \\")
    print("    output_dir=data/tokenized_datasets/mimic \\")
    print("    out_fn=train")
    print()
    
    print("2. If the process is interrupted, restart with resume:")
    print("ethos_tokenize -m worker='range(0,7)' \\")
    print("    input_dir=data/mimic-2.2-meds/data/train \\")
    print("    output_dir=data/tokenized_datasets/mimic \\")
    print("    out_fn=train \\")
    print("    resume=true")
    print()
    
    print("3. For test data (using vocab from train):")
    print("ethos_tokenize -m worker='range(0,2)' \\")
    print("    input_dir=data/mimic-2.2-meds/data/test \\")
    print("    vocab=data/tokenized_datasets/mimic/train \\")
    print("    output_dir=data/tokenized_datasets/mimic \\")
    print("    out_fn=test \\")
    print("    resume=true")
    print()
    
    print("=== How it works ===")
    print("- When resume=true, the system checks for a checkpoint file")
    print("- If found, it loads the list of completed stages")
    print("- Skips already completed stages and continues from where it left off")
    print("- Saves progress after each stage completion")
    print("- Checkpoint file: output_dir/tokenization_checkpoint.json")
    print()
    
    print("=== Benefits ===")
    print("- No need to restart from the beginning if interrupted")
    print("- Saves time and computational resources")
    print("- Handles long-running tokenization jobs gracefully")
    print("- Works with both single and multi-worker configurations")

if __name__ == "__main__":
    test_tokenization_restart()
