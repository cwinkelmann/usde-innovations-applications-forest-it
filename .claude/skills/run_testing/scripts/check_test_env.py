#!/usr/bin/env python3
"""
HILDA Test Environment Checker
Validates test environment readiness for HILDA test suite.
"""
import subprocess
import os
from pathlib import Path


def check_test_environment():
    """Check HILDA test environment readiness"""
    issues = []

    # Check conda environment
    try:
        result = subprocess.run(['conda', 'info'], capture_output=True, text=True)
        if 'active environment' not in result.stdout.lower():
            issues.append("❌ Conda environment not active")
        else:
            print("✅ Conda environment active")
    except:
        issues.append("❌ Conda not available")

    # Check test data
    test_data_dir = Path("tests/data")
    required_paths = [
        test_data_dir / "Isabela/ISWF01_22012023_subset",
        test_data_dir / "raw_images/mavic2pro",
        test_data_dir / "raw_images/matrice4e"
    ]

    for path in required_paths:
        if path.exists():
            image_count = len(list(path.glob("**/*.JPG")))
            print(f"✅ {path.name}: {image_count} images")
        else:
            issues.append(f"❌ Missing test data: {path}")

    # Check GPU availability
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        print(f"🔧 GPU: {'✅ Available' if gpu_available else '❌ CPU only'}")
    except:
        issues.append("❌ PyTorch not available")

    # Check models
    models_dir = test_data_dir / "models"
    if models_dir.exists():
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        print(f"🤖 Models: {len(model_dirs)} available ({', '.join(d.name for d in model_dirs)})")
    else:
        issues.append("❌ No test models found")

    return issues, gpu_available


def validate_test_data():
    """Validate test data integrity"""
    test_data_dir = Path("tests/data")

    # Check image counts
    isabela_images = list((test_data_dir / "Isabela/ISWF01_22012023_subset").glob("*.JPG"))
    print(f"📸 Isabela: {len(isabela_images)} images (expect 9)")

    mavic_images = list((test_data_dir / "raw_images/mavic2pro").glob("**/*.JPG"))
    print(f"📸 Mavic 2 Pro: {len(mavic_images)} images (expect 50)")

    matrice_images = list((test_data_dir / "raw_images/matrice4e").glob("**/*.JPG"))
    print(f"📸 Matrice 4E: {len(matrice_images)} images (expect 51)")

    # Check model files
    models_dir = test_data_dir / "models"
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                model_file = next(model_dir.glob("*.pth"), None)
                config_file = next(model_dir.glob("**/config.yaml"), None)
                status = "✅" if (model_file and config_file) else "❌"
                print(f"🤖 {model_dir.name}: {status}")

    return len(isabela_images), len(mavic_images), len(matrice_images)


if __name__ == "__main__":
    print("🔍 HILDA Test Environment Check\n")

    issues, gpu_available = check_test_environment()

    if issues:
        print("\n⚠️ Environment Issues:")
        for issue in issues:
            print(f"  {issue}")
        print("\n💡 Some tests may be skipped due to missing dependencies")
    else:
        print("\n🎉 Test environment ready!")

    print("\n📊 Test Data Validation:")
    validate_test_data()