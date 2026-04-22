import subprocess
import os
import time
import sys

def run_script(dir_path, script_name):
    print(f"\n--- Running {script_name} in {dir_path} ---")
    try:
        # Run the script from its directory
        process = subprocess.Popen(
            [sys.executable, script_name],
            cwd=os.path.abspath(dir_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Read output in real-time
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        if process.returncode == 0:
            print(f"--- Finished {script_name} successfully ---")
        else:
            print(f"--- {script_name} failed with code {process.returncode} ---")
    except Exception as e:
        print(f"Error running {script_name}: {e}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Start Flask App in background
    print("\n--- Launching Classification Web App ---")
    flask_app_path = os.path.join(base_dir, "Classification", "app.py")
    flask_process = subprocess.Popen(
        [sys.executable, flask_app_path],
        cwd=os.path.join(base_dir, "Classification")
    )
    print("Flask Server started. Point your browser to http://127.0.0.1:5000")
    print("Waiting 5 seconds for server to initialize...\n")
    time.sleep(5)
    
    # 2. Run Segmentation Modules
    segmentation_tasks = [
        ("Blood Vessel Segmentation", "test.py"),
        ("Haemorage Segmentation", "test.py"),
        ("Hard Exudate Segmentation", "test.py"),
        ("Microanuerism Segmentation", "test.py"),
        ("Optical Disc Segmentation", "test.py"),
        ("Soft Exudate Segmentation", "test.py"),
    ]
    
    for folder, script in segmentation_tasks:
        run_script(os.path.join(base_dir, folder), script)
    
    print("\n" + "="*50)
    print("All tasks completed.")
    print("The Flask server is still running. Press Ctrl+C to stop it (or manually kill the process).")
    print("="*50)
    
    # Keep the main process alive to maintain the Flask server until interrupted
    try:
        flask_process.wait()
    except KeyboardInterrupt:
        print("\nStopping Flask server...")
        flask_process.terminate()

if __name__ == "__main__":
    main()
