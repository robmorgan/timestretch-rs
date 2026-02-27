#!/usr/bin/env python3
import json
import os
import subprocess
import sys

def main():
    repo_root = os.getcwd()
    manifest_path = os.path.join(repo_root, "test_manifest.json")
    output_dir = os.path.join(repo_root, "optimize/outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest {manifest_path} not found.")
        sys.exit(1)
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    print(f"Running timestretch-rs against {len(manifest)} test cases...")
    
    # Ensure binary is built
    subprocess.run(["cargo", "build", "--release", "--features", "cli"], check=True)
    binary = os.path.join(repo_root, "target/release/timestretch-cli")
    
    for item in manifest:
        source = os.path.join(repo_root, "optimize", item['source'])
        ratio = item['ratio']
        source_base = os.path.basename(item['source']).replace('.wav', '')
        output = os.path.join(output_dir, f"{source_base}_test_{ratio}.wav")
        
        if not os.path.exists(source):
            print(f"Warning: Source file not found: {source}")
            continue
            
        print(f"Processing {item['description']} (ratio: {ratio})...")
        
        cmd = [binary, source, output, "--ratio", str(ratio)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {source}: {e.stderr.decode()}")
            
    print("Test suite completed.")

if __name__ == "__main__":
    main()
