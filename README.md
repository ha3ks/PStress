# CPU Stress Tester

A Python-based stress testing tool for both CPUs and GPUs with real-time performance monitoring. Test your CPU with intensive calculations showing MFLOPS/GFLOPS, or stress test your GPU with matrix operations displaying TFLOPS.

## Features

- **Multi-core CPU support**: Utilize all or a specific number of CPU cores
- **GPU support**: CUDA-based GPU stress testing with PyTorch or CuPy
- **Configurable duration**: Run tests for any length of time or continuously
- **Real-time FLOPS monitoring**: See performance metrics updated every second
- **Colorized output**: Easy-to-read color-coded performance data
- **Continuous mode**: Run indefinitely until manually stopped
- **Cross-platform**: Works on Windows, Linux, and macOS
- **Minimal dependencies**: CPU mode uses only Python standard library

## Requirements

**For CPU testing:**
- Python 3.6 or higher

**For GPU testing:**
- Python 3.6 or higher
- NVIDIA GPU with CUDA support
- One of the following:
  - PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
  - CuPy: `pip install cupy-cuda11x` (replace 11x with your CUDA version)

### Basic Usage

**CPU Testing** (default):
Run with default settings (all cores, 60 seconds):
```bash
python cpu_stress_test.py
```

**GPU Testing:**
```bash
python cpu_stress_test.py --mode gpu -d 60
```

Run continuously until stopped with Ctrl+C:
```bash
python cpu_stress_test.py -d 0
```

### Advanced Options

### Advanced Options

### Command-line Arguments

- `-c, --cores <number>`: Number of CPU cores to use (default: all available) - CPU mode only
- `-d, --duration <seconds>`: Duration in seconds to run the test (default: 60, use 0 for continuous)
- `-v, --verbose`: Show per-core performance breakdown (CPU) or detailed GPU info
- `--mode <cpu|gpu>`: Test mode - 'cpu' for CPU testing (default) or 'gpu' for GPU testing
- `-h, --help`: Show help message 2

## Examples

**Test all cores for 1 minute:**
```bash
python cpu_stress_test.py
```

**Extended stress test (10 minutes):**
```bash
python cpu_stress_test.py -d 600
```

**Continuous mode (run until Ctrl+C):**
```bash
python cpu_stress_test.py -d 0
```
**Verbose mode with per-core breakdown:**
```bash
python cpu_stress_test.py -v -d 30
```

## How It Works

**CPU Mode:**
The stress tester spawns multiple worker processes (one per CPU core by default) that perform intensive mathematical calculations including:

- Square root and power operations
- Factorial calculations
- List comprehensions with exponential operations

Each worker runs continuously for the specified duration (or indefinitely in continuous mode), reporting real-time FLOPS (Floating Point Operations Per Second) performance metrics. The display shows:

- **Current FLOPS/s**: Instantaneous performance rate
- **Cumulative FLOPS**: Total operations performed
- **Color-coded metrics**: Performance scaled automatically (FLOPS → KFLOPS → MFLOPS → GFLOPS)

**GPU Mode:**
The GPU stress tester performs large matrix multiplications (4096x4096) using CUDA:

- Uses PyTorch or CuPy for GPU acceleration
- Each iteration performs 2*N³ floating-point operations
- Reports performance in TFLOPS (Teraflops)
- Fully utilizes GPU compute capabilities
- Monitors GPU temperature and utilization (with verbose mode)
The stress tester spawns multiple worker processes (one per CPU core by default) that perform intensive mathematical calculations including:
## Use Cases

- **Thermal testing**: Check CPU/GPU cooling performance under load
- **Stability testing**: Verify system stability after overclocking
- **Performance benchmarking**: Compare CPU/GPU performance across systems
- **Burn-in testing**: Stress test new hardware
- **Power consumption testing**: Monitor power draw under full load

## Warning

This tool will push your CPU/GPU to 100% utilization. Monitor your system temperatures and ensure adequate cooling. Not recommended for extended periods on laptops or systems with insufficient cooling.
- Square root and power operations
- Factorial calculations
- List comprehensions with exponential operations

Each worker runs continuously for the specified duration, reporting its progress when complete.

## Stopping the Test

Press `Ctrl+C` to interrupt the test at any time.

## Use Cases

- **Thermal testing**: Check CPU cooling performance under load
- **Stability testing**: Verify system stability after overclocking
- **Performance benchmarking**: Compare CPU performance across systems
- **Burn-in testing**: Stress test new hardware

## Warning

This tool will push your CPU to 100% utilization. Monitor your system temperatures and ensure adequate cooling. Not recommended for extended periods on laptops or systems with insufficient cooling.

This project uses some Vibe Coded atributes (mainly math as I suck at math) and I cannot test the GPU option as I have integrated graphics.
