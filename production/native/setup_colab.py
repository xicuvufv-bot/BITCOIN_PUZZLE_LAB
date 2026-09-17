#!/usr/bin/env python3
"""
setup_colab.py - Python helper for Google Colab automation
Usage in Colab:
    %run setup_colab.py
"""
import subprocess
import os
import sys
import torch
import json
import time
import threading
import signal
import os

def detect_gpu():
    """Detect GPU and return (name, cc, arch, recommended_puzzle, budget)"""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available! Enable GPU in Colab.")
    
    gpu_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    cc = f"{cc_major}.{cc_minor}"
    
    arch_map = {
        '7.5': 'sm_75',   # T4
        '8.0': 'sm_80',   # A100 40GB
        '8.6': 'sm_86',   # A100 80GB
        '8.9': 'sm_89',   # H100
        '9.0': 'sm_90',   # H100
    }
    arch = arch_map.get(f'{cc_major}.{cc_minor}', 'sm_86')
    
    # Puzzle recommendations by GPU
    puzzle_map = {
        (7, 5): (135, 30),   # T4: Puzzle #135 (R2, 66-bit)
        (8, 0): (140, 32),   # A100 40GB
        (8, 6): (145, 33),   # A100 80GB
        (8, 9): (150, 34),   # H100
        (9, 0): (150, 34),   # H100
    }
    puzzle, budget = puzzle_map.get((cc_major, cc_minor), (135, 30))
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return {
        'name': gpu_name,
        'cc': cc,
        'arch': arch,
        'puzzle': puzzle,
        'budget': budget,
        'memory_gb': gpu_mem
    }

def mount_drive():
    """Mount Google Drive"""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        print("✅ Google Drive mounted")
        return True
    except Exception as e:
        print(f"⚠️ Drive mount failed: {e}")
        return False

def install_deps():
    """Install system dependencies"""
    subprocess.run(['apt-get', 'update', '-qq'], capture_output=True)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 
                    'cmake', 'build-essential', 'git', 'wget'], 
                   capture_output=True)
    print("✅ Dependencies installed")

def build_project(arch):
    """Build the GLV Kangaroo engine"""
    os.chdir('/content/BITCOIN_PUZZLE_LAB/production/native')
    cmd = [
        'nvcc', '-O3', f'-arch={arch}', '-Xcompiler=/O2', '-Xptxas', '-O3',
        '-I.', '-std=c++17', '-o', 'kangaroo_glv_gpu',
        'kangaroo_glv_gpu.cu', '-lcudart'
    ]
    print(f"Building with: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"❌ Build failed:\n{result.stderr}")
        return False
    print("✅ Build successful")
    return True

def run_sanity_test():
    """Run the -test sanity check"""
    print("\n🧪 Running Sanity Test (-test)...")
    result = subprocess.run(['./kangaroo_glv_gpu', '-test'], 
                          capture_output=True, text=True, timeout=120)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    return result.returncode == 0

def get_checkpoint_path(puzzle):
    """Get checkpoint path on Google Drive"""
    checkpoint_dir = '/content/drive/MyDrive/Kangaroo_Checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    return f'{checkpoint_dir}/puzzle_{puzzle}.work'

def run_solver(puzzle, budget, checkpoint_path):
    """Launch the solver with checkpoint sync"""
    cmd = [
        './kangaroo_glv_gpu',
        f'-puzzle {puzzle}',
        '-gpu 0',
        f'-checkpoint {checkpoint_path}',
        '-dpbits 26',
        f'-budget {budget}',
        '-sleep 300'
    ]
    cmd_str = ' '.join(cmd)
    print(f"🚀 Launching: {cmd_str}")
    
    proc = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True, 
        bufsize=1
    )
    
    def stream_output():
        for line in proc.stdout:
            print(line.rstrip())
            if 'FOUND' in line or 'COLLISION' in line:
                print(f"🎉 SUCCESS: {line.strip()}")
    
    thread = threading.Thread(target=stream_output, daemon=True)
    thread.start()
    
    try:
        while thread.is_alive():
            time.sleep(30)
            if proc.poll() is not None:
                break
    except KeyboardInterrupt:
        print('\n🛑 Interrupted, saving checkpoint...')
        proc.terminate()
        proc.wait()
        print('Checkpoint saved.')
    
    return proc.wait() == 0

def main():
    print("=" * 60)
    print("🚀 GLV Kangaroo - Colab Deployment")
    print("=" * 60)
    
    # 1. Detect GPU
    gpu_info = detect_gpu()
    print(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f} GB)")
    print(f"Compute Capability: {gpu_info['cc']} -> ARCH: {gpu_info['arch']}")
    
    # 2. Mount Drive
    mount_drive()
    
    # 3. Install deps
    install_deps()
    
    # 3. Clone repo
    repo_dir = '/content/BITCOIN_PUZZLE_LAB'
    if not os.path.exists(repo_dir):
        print("📥 Cloning repository...")
        subprocess.run(['git', 'clone', 'https://github.com/JeanLucPons/Kangaroo.git', 
                       f'{repo_dir}_orig'], capture_output=True)
        subprocess.run(['cp', '-r', f'{repo_dir}_orig', repo_dir], capture_output=True)
    else:
        subprocess.run(['git', '-C', repo_dir, 'pull'], capture_output=True)
    
    os.chdir(f'{repo_dir}/production/native')
    print(f"Working in: {os.getcwd()}")
    
    # 4. Build
    if not build_project(gpu_info['arch']):
        sys.exit(1)
    
    # 5. Sanity test
    if not run_sanity_test():
        print("❌ Sanity test failed!")
        sys.exit(1)
    
    # 6. Get puzzle recommendation
    puzzle = gpu_info['puzzle']
    budget = gpu_info['budget']
    print(f"\n🎯 Recommended: Puzzle #{puzzle} (Budget: 2^{budget})")
    
    # 7. Checkpoint path
    checkpoint_path = get_checkpoint_path(gpu_info['puzzle'])
    print(f"Checkpoint: {checkpoint_path}")
    
    # 8. Launch
    print("\n" + "="*60)
    print("🚀 Launching solver...")
    print("="*60)
    run_solver(gpu_info['puzzle'], gpu_info['budget'], 
               f'/content/drive/MyDrive/Kangaroo_Checkpoints/puzzle_{gpu_info["puzzle"]}.work')

if __name__ == '__main__':
    main()