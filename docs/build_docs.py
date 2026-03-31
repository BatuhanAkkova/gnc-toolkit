import os
import shutil
import subprocess

def build():
    # 1. Skip tutorial copying as they are already in docs/tutorials/
    print("Detected tutorials directory in docs, skipping copy step...")
    
    # 2. Run sphinx-apidoc to generate API structure
    print("Running sphinx-apidoc...")
    subprocess.run(["sphinx-apidoc", "-o", "api", os.path.abspath("../src/opengnc"), "-f"], check=True)
    
    # 3. Run sphinx-build to generate HTML output
    print("Running sphinx-build...")
    subprocess.run(["sphinx-build", "-b", "html", ".", "_build/html"], check=True)
    print("Build completed successfully!")

if __name__ == "__main__":
    # Ensure working directory is the docs folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    build()




