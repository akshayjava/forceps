#!/usr/bin/env python3
import shutil, subprocess, sys, os
APP_NAME = "FORCEPS"
ENTRY = "app/main.py"

def clean():
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("dist", ignore_errors=True)
    for f in os.listdir("."):
        if f.endswith(".spec"):
            os.remove(f)

def build():
    print("[build] Running PyInstaller...")
    subprocess.run([sys.executable, "-m", "PyInstaller", "--onefile", "--name", APP_NAME, ENTRY], check=True)
    print("[build] done.")

def make_zip():
    zipname = f"{APP_NAME}_package.zip"
    if os.path.exists(zipname):
        os.remove(zipname)
    shutil.make_archive(APP_NAME + "_package", 'zip', ".")
    print("[zip] created", zipname)

if __name__ == "__main__":
    clean()
    build()
    make_zip()
