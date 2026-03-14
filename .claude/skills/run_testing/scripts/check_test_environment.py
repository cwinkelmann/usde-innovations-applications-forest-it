#!/usr/bin/env python3
"""
Check HILDA test environment readiness
"""
import subprocess
import sys
from pathlib import Path


def check_conda_environment():
    """Check if conda environment is active and which one"""
    try:
        result = subprocess.run(['conda', 'info'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'active environment' in line:
                    env_name = line.split(':')[-1].strip()
                    print(f"✅ Conda environment: {env_name}")
                    return True, env_name
        print("❌ Conda environment not active")
        return False, None
    except FileNotFoundError:
        print("❌ Conda not found in PATH")
        return False, None


def check_gpu_availability():
    """Check if GPU is available for model tests"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0

        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU: {gpu_name} ({gpu_count} device{'s' if gpu_count > 1 else ''})")
        else:
            print("❌ GPU: Not available (CPU only)")

        return gpu_available, gpu_count
    except ImportError:
        print("❌ PyTorch not installed - cannot check GPU")
        return False, 0


def check_test_data():
    """Check test data availability and integrity"""
    project_root = Path(__file__).parent.parent.parent.parent
    test_data_dir = project_root / "tests" / "data"

    test_data_status = {}

    # Required test data paths
    test_paths = {
        "Isabela subset": test_data_dir / "Isabela" / "ISWF01_22012023_subset",
        "Mavic 2 Pro": test_data_dir / "raw_images" / "mavic2pro",
        "Matrice 4E": test_data_dir / "raw_images" / "matrice4e",
        "Models": test_data_dir / "models",
        "Geospatial": test_data_dir / "geospatial"
    }

    total_images = 0
    for name, path in test_paths.items():
        if path.exists():
            # Count images if it's an image directory
            if name in ["Isabela subset", "Mavic 2 Pro", "Matrice 4E"]:
                images = list(path.glob("**/*.JPG")) + list(path.glob("**/*.jpg"))
                total_images += len(images)
                print(f"✅ {name}: {len(images)} images")
                test_data_status[name] = {"present": True, "count": len(images)}
            elif name == "Models":
                model_dirs = [d for d in path.iterdir() if d.is_dir()]
                print(f"✅ {name}: {len(model_dirs)} models ({', '.join(d.name for d in model_dirs)})")
                test_data_status[name] = {"present": True, "count": len(model_dirs)}
            else:
                files = list(path.glob("*"))
                print(f"✅ {name}: {len(files)} files")
                test_data_status[name] = {"present": True, "count": len(files)}
        else:
            print(f"❌ {name}: Not found at {path}")
            test_data_status[name] = {"present": False, "count": 0}

    print(f"\n📊 Total test images: {total_images}")
    return test_data_status


def check_python_dependencies():
    """Check key Python dependencies for HILDA"""
    required_packages = [
        'geopandas', 'pandas', 'numpy', 'pytest', 'pathlib',
        'rasterio', 'loguru', 'hydra-core', 'torch'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)

    return missing_packages


def estimate_test_runtimes():
    """Provide test runtime estimates based on environment"""
    print("\n⏱️ Estimated Test Runtimes:")
    print("Fast unit tests (metadata, file listing):     10-30 seconds")
    print("Image database tests (9-50 images):          30 seconds - 2 minutes")
    print("Mission import tests (100+ images):          1-3 minutes")
    print("Model inference tests (GPU required):        2-10 minutes")
    print("Full test suite:                             5-15 minutes")


def main():
    """Main environment check"""
    print("🔍 HILDA Test Environment Check")
    print("=" * 50)

    # Check conda
    conda_ok, env_name = check_conda_environment()

    # Check GPU
    gpu_ok, gpu_count = check_gpu_availability()

    # Check test data
    print("\n📁 Test Data Status:")
    test_data_status = check_test_data()

    # Check Python dependencies
    print("\n📦 Python Dependencies:")
    missing_deps = check_python_dependencies()

    # Runtime estimates
    estimate_test_runtimes()

    # Summary
    print("\n📋 Environment Summary:")
    if conda_ok:
        print(f"✅ Conda environment active: {env_name}")
    else:
        print("❌ Conda environment setup needed")

    if gpu_ok:
        print(f"✅ GPU available for model tests ({gpu_count} device{'s' if gpu_count > 1 else ''})")
    else:
        print("⚠️ CPU only - model tests will be skipped")

    missing_data = [name for name, status in test_data_status.items() if not status['present']]
    if missing_data:
        print(f"❌ Missing test data: {', '.join(missing_data)}")
    else:
        print("✅ All test data present")

    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("   Install with: pip install " + " ".join(missing_deps))
    else:
        print("✅ All dependencies available")

    # Recommendations
    print("\n💡 Recommendations:")
    if not conda_ok:
        print("   - Activate conda environment with: conda activate [env_name]")

    if missing_data:
        print("   - Download missing test data to tests/data/")
        print("   - Some tests will be skipped without required data")

    if not gpu_ok:
        print("   - Run on GPU machine for full model testing")
        print("   - Or run with: pytest tests/ -k 'not herdnet_predict'")

    return {
        'conda': conda_ok,
        'gpu': gpu_ok,
        'test_data': test_data_status,
        'dependencies': len(missing_deps) == 0
    }


if __name__ == "__main__":
    status = main()

    # Exit code based on critical issues
    if not status['conda'] or not status['dependencies']:
        sys.exit(1)
    else:
        sys.exit(0)