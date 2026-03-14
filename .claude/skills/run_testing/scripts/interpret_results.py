#!/usr/bin/env python3
"""
HILDA Test Result Interpreter
Parses pytest output and provides actionable feedback.
"""


def interpret_test_results(return_code, output):
    """Interpret pytest results and provide guidance"""

    if return_code == 0:
        print("🎉 ALL TESTS PASSED!")
        return

    # Extract failure information
    if "FAILED" in output:
        print("❌ SOME TESTS FAILED")

        # Common failure patterns and solutions
        if "FileNotFoundError" in output:
            print("💡 Solution: Check test data is present in tests/data/")
            print("   Run: ls tests/data/ to verify")

        if "ImportError" in output or "ModuleNotFoundError" in output:
            print("💡 Solution: Install HILDA package in development mode:")
            print("   Run: pip install -e .")

        if "CUDA" in output or "GPU" in output:
            print("💡 Solution: Run on GPU machine or skip model tests")

        if "permission" in output.lower():
            print("💡 Solution: Check file permissions in test data directory")

    if "ERROR" in output:
        print("❌ TEST SETUP ERRORS")
        print("💡 Check pytest configuration and imports")

    # Extract specific failed tests
    failed_tests = []
    for line in output.split('\n'):
        if '::' in line and 'FAILED' in line:
            failed_tests.append(line.split()[0])

    if failed_tests:
        print(f"\n📋 Failed tests ({len(failed_tests)}):")
        for test in failed_tests[:5]:  # Show first 5
            print(f"   {test}")
        if len(failed_tests) > 5:
            print(f"   ... and {len(failed_tests) - 5} more")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        return_code = int(sys.argv[1])
        output = sys.argv[2]
        interpret_test_results(return_code, output)
    else:
        print("Usage: python interpret_results.py <return_code> <output>")