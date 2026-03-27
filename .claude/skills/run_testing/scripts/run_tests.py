#!/usr/bin/env python3
"""
HILDA Test Runner - Execute different test categories
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return results with timing"""
    print(f"\n🏃 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    print(f"\n⏱️ Completed in {duration:.1f} seconds")
    return result.returncode, result.stdout, result.stderr, duration


def run_fast_tests():
    """Run fast unit tests only"""
    cmd = [
        "pytest",
        "tests/test_image_database.py::TestListImages",
        "tests/test_image_database.py::TestMetadataExtraction",
        "-v", "--tb=short"
    ]
    return run_command(cmd, "Fast Unit Tests (Metadata & File Listing)")


def run_integration_tests():
    """Run integration tests"""
    cmd = [
        "pytest",
        "tests/test_image_database.py::TestCreateImageDb",
        "tests/pipeline/mission_import/",
        "-v", "--tb=short"
    ]
    return run_command(cmd, "Integration Tests (Database & Mission Import)")


def run_model_tests():
    """Run model inference tests (requires GPU)"""
    cmd = [
        "pytest",
        "tests/test_herdnet_predict.py",
        "tests/test_geospatial_prediction.py",
        "-v", "--tb=short"
    ]
    return run_command(cmd, "Model Tests (HerdNet & Geospatial Prediction)")


def run_specific_test(test_path):
    """Run a specific test file or test method"""
    cmd = ["pytest", test_path, "-v", "--tb=short"]
    return run_command(cmd, f"Specific Test: {test_path}")


def run_all_tests():
    """Run the complete test suite"""
    cmd = ["pytest", "tests/", "-v", "--tb=short", "--durations=10"]
    return run_command(cmd, "Complete Test Suite")


def run_mission_import_tests():
    """Run mission import pipeline tests only"""
    cmd = ["pytest", "tests/pipeline/mission_import/", "-v", "--tb=short"]
    return run_command(cmd, "Mission Import Pipeline Tests")


def run_debug_mode(test_path):
    """Run tests in debug mode with maximum verbosity"""
    cmd = ["pytest", test_path, "-v", "-s", "--tb=long", "--capture=no"]
    return run_command(cmd, f"Debug Mode: {test_path}")


def interpret_results(return_code, stdout, stderr, duration):
    """Interpret and summarize test results"""
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)

    if return_code == 0:
        print("🎉 ALL TESTS PASSED!")

        # Extract test count from output
        lines = stdout.split('\n')
        for line in lines:
            if 'passed' in line and ('warning' in line or 'error' in line or '==' in line):
                print(f"✅ {line.strip()}")
                break

    else:
        print("❌ SOME TESTS FAILED")

        # Extract failure information
        failed_tests = []
        errors = []

        for line in stdout.split('\n'):
            if 'FAILED' in line and '::' in line:
                failed_tests.append(line.strip())
            elif 'ERROR' in line and '::' in line:
                errors.append(line.strip())

        if failed_tests:
            print(f"\n📋 Failed Tests ({len(failed_tests)}):")
            for test in failed_tests[:10]:  # Show first 10
                print(f"   ❌ {test}")
            if len(failed_tests) > 10:
                print(f"   ... and {len(failed_tests) - 10} more")

        if errors:
            print(f"\n🚨 Test Errors ({len(errors)}):")
            for error in errors[:5]:
                print(f"   🚨 {error}")

        # Provide troubleshooting guidance
        print("\n💡 TROUBLESHOOTING GUIDE:")

        if "FileNotFoundError" in stdout or "No such file" in stdout:
            print("   📁 Missing test data - run: python scripts/check_test_environment.py")

        if "ImportError" in stdout or "ModuleNotFoundError" in stdout:
            print("   📦 Missing dependencies - run: pip install -e .")

        if "CUDA" in stdout or "GPU" in stdout:
            print("   🔧 GPU issues - try: CUDA_VISIBLE_DEVICES='' pytest [test]")

        if "permission" in stdout.lower():
            print("   🔒 Permission issues - check file permissions in tests/data/")

        if "EPSG" in stdout or "CRS" in stdout or "coordinate" in stdout.lower():
            print("   🗺️ Coordinate system issues - verify pyproj/GDAL installation")

    print(f"\n⏱️ Total runtime: {duration:.1f} seconds")

    # Performance feedback
    if duration < 30:
        print("⚡ Fast execution")
    elif duration < 300:  # 5 minutes
        print("🐌 Normal execution time")
    else:
        print("🐢 Slow execution - consider running smaller test subsets")


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(
        description="HILDA Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Categories:
  fast          Run fast unit tests only (10-30 seconds)
  integration   Run integration tests (1-3 minutes)
  models        Run model tests - requires GPU (2-10 minutes)
  mission       Run mission import tests (1-3 minutes)
  all           Run complete test suite (5-15 minutes)

Examples:
  python run_tests.py fast
  python run_tests.py integration
  python run_tests.py specific tests/test_image_database.py::TestListImages
  python run_tests.py debug tests/test_herdnet_predict.py
        """
    )

    parser.add_argument(
        'test_type',
        choices=['fast', 'integration', 'models', 'mission', 'all', 'specific', 'debug'],
        help='Type of tests to run'
    )

    parser.add_argument(
        'test_path',
        nargs='?',
        help='Specific test path (required for specific/debug modes)'
    )

    parser.add_argument(
        '--env-check',
        action='store_true',
        help='Run environment check before testing'
    )

    args = parser.parse_args()

    # Change to project root
    project_root = Path(__file__).parent.parent.parent.parent
    print(f"📁 Working directory: {project_root}")

    # Optional environment check
    if args.env_check:
        print("🔍 Running environment check first...")
        env_result = subprocess.run([
            sys.executable,
            str(project_root / "testing-skill/scripts/check_test_environment.py")
        ])
        if env_result.returncode != 0:
            print("⚠️ Environment issues detected - continuing anyway...")

    # Execute requested test category
    if args.test_type == 'fast':
        return_code, stdout, stderr, duration = run_fast_tests()

    elif args.test_type == 'integration':
        return_code, stdout, stderr, duration = run_integration_tests()

    elif args.test_type == 'models':
        return_code, stdout, stderr, duration = run_model_tests()

    elif args.test_type == 'mission':
        return_code, stdout, stderr, duration = run_mission_import_tests()

    elif args.test_type == 'all':
        return_code, stdout, stderr, duration = run_all_tests()

    elif args.test_type == 'specific':
        if not args.test_path:
            print("❌ Error: test_path required for 'specific' mode")
            return 1
        return_code, stdout, stderr, duration = run_specific_test(args.test_path)

    elif args.test_type == 'debug':
        if not args.test_path:
            print("❌ Error: test_path required for 'debug' mode")
            return 1
        return_code, stdout, stderr, duration = run_debug_mode(args.test_path)

    # Interpret and display results
    interpret_results(return_code, stdout, stderr, duration)

    return return_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)