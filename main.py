
import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("SIRLO: Digital Twin Satellite Constellation Framework")
    print("Initializing environment...")

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available. Running on CPU.")
    except ImportError:
        print("PyTorch not installed.")

    try:
        import ee
        print("Earth Engine API installed.")
    except ImportError:
        print("Earth Engine API not installed.")

    try:
        import skyfield.api
        print(f"Skyfield version: {skyfield.api.__version__}")
    except ImportError:
        print("Skyfield not installed.")

if __name__ == "__main__":
    main()
