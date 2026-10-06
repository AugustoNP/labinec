#!/usr/bin/env python3
import os
import subprocess
import sys
import venv
from pathlib import Path

VENV_DIR = Path(".venv")

def create_venv():
    print(f"[*] Creating virtual environment in '{VENV_DIR}'...")
    builder = venv.EnvBuilder(with_pip=True, clear=True)
    builder.create(VENV_DIR)

def get_venv_python():
    """Returns the correct Python executable path based on the OS."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def install_requirements(python_exe):
    print("[*] Upgrading pip...")
    subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    print("[*] Installing dependencies from requirements.txt...")
    subprocess.run([str(python_exe), "-m", "pip", "install", "-r", "requirements.txt"], check=True)

def main():
    if not Path("requirements.txt").exists():
        print("[!] Error: requirements.txt not found in current directory.")
        sys.exit(1)

    create_venv()
    python_exe = get_venv_python()
    install_requirements(python_exe)

    print("\n[+] Setup complete! To activate your environment, run:")
    if sys.platform == "win32":
        print(f"    {VENV_DIR}\\Scripts\\activate")
    else:
        print(f"    source {VENV_DIR}/bin/activate")

if __name__ == "__main__":
    main()
