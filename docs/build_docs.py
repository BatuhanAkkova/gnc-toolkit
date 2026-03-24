import os
import shutil
import subprocess

def build():
    # 1. Copy tutorials into docs directory
    src_tutorials = os.path.abspath("../tutorials")
    dest_tutorials = os.path.abspath("tutorials")
    print(f"Copying {src_tutorials} to {dest_tutorials}...")
    if os.path.exists(dest_tutorials):
        shutil.rmtree(dest_tutorials)
    shutil.copytree(src_tutorials, dest_tutorials)
    
    # 2. Run sphinx-apidoc to generate API structure
    print("Running sphinx-apidoc...")
    subprocess.run(["sphinx-apidoc", "-o", "api", os.path.abspath("../src/gnc_toolkit"), "-f"], check=True)
    
    # 3. Run sphinx-build to generate HTML output
    print("Running sphinx-build...")
    subprocess.run(["sphinx-build", "-b", "html", ".", "_build/html"], check=True)
    print("Build completed successfully!")

if __name__ == "__main__":
    # Ensure working directory is the docs folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    build()
