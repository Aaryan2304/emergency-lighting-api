#!/usr/bin/env python3
"""
Debug script to test Poppler installation on Render.
This will help troubleshoot PDF conversion issues.
"""

import os
import platform
import subprocess
import sys

def check_poppler_installation():
    """Check if Poppler is properly installed."""
    print("=== Poppler Installation Debug ===")
    print(f"Platform: {platform.system()}")
    print(f"Environment: {'Render' if os.getenv('RENDER') else 'Local'}")
    
    # Check common Poppler locations
    poppler_paths = [
        "/usr/bin",
        "/usr/local/bin", 
        "/opt/poppler/bin"
    ]
    
    print("\n=== Checking Poppler Paths ===")
    for path in poppler_paths:
        pdftoppm_path = os.path.join(path, "pdftoppm")
        exists = os.path.exists(pdftoppm_path)
        print(f"{pdftoppm_path}: {'✓ EXISTS' if exists else '✗ NOT FOUND'}")
    
    # Try to run poppler commands
    print("\n=== Testing Poppler Commands ===")
    commands = ["pdftoppm", "pdftocairo", "pdfinfo"]
    
    for cmd in commands:
        try:
            result = subprocess.run([cmd, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"{cmd}: ✓ WORKING - {result.stdout.strip()}")
            else:
                print(f"{cmd}: ✗ FAILED - Return code {result.returncode}")
        except FileNotFoundError:
            print(f"{cmd}: ✗ NOT FOUND")
        except subprocess.TimeoutExpired:
            print(f"{cmd}: ✗ TIMEOUT")
        except Exception as e:
            print(f"{cmd}: ✗ ERROR - {e}")
    
    # Check PATH
    print(f"\n=== PATH ===")
    path_env = os.getenv('PATH', '')
    for path in path_env.split(':'):
        if 'poppler' in path.lower() or any(os.path.exists(os.path.join(path, cmd)) for cmd in commands):
            print(f"✓ {path}")
    
    print("\n=== PDF2Image Test ===")
    try:
        from pdf2image import convert_from_path
        print("✓ pdf2image imported successfully")
        
        # Try to get poppler version through pdf2image
        try:
            # This should fail gracefully if poppler isn't working
            test_result = subprocess.run(["pdftoppm", "-v"], 
                                       capture_output=True, text=True, timeout=5)
            print(f"✓ pdf2image can access poppler: {test_result.stderr.strip() if test_result.stderr else 'OK'}")
        except Exception as e:
            print(f"✗ pdf2image cannot access poppler: {e}")
            
    except ImportError as e:
        print(f"✗ pdf2image import failed: {e}")

if __name__ == "__main__":
    check_poppler_installation()
