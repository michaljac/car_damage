"""
Inference script with CarROICrop (on-the-fly)

TODO: Implement inference with trained model + CarROICrop
This will use the trained model and apply CarROICrop on-the-fly during inference.

For now, use tests/uni_preprocess.py to test CarROICrop functionality.
"""

import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../mmdetection'))


def main():
    """Inference with CarROICrop - to be implemented after training."""
    print("Inference will be implemented after training is complete.")
    print("For now, use tests/uni_preprocess.py to test CarROICrop functionality.")


if __name__ == '__main__':
    main()
