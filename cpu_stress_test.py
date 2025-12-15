#!/usr/bin/env python3
"""
CPU Stress Tester
A multi-threaded CPU stress testing tool with configurable parameters.
A mix of vibe code and Python knoweldge
"""

import argparse
import multiprocessing
import time
import sys
import math
from datetime import datetime

# Try to import GPU libraries
GPU_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_LIBRARY = "pytorch"
except ImportError:
    try:
        import cupy as cp
        GPU_AVAILABLE = True
        GPU_LIBRARY = "cupy"
    except ImportError:
        GPU_LIBRARY = None


# ANSI Color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Regular colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    # Background colors
    BG_RED = "\033[101m"
    BG_GREEN = "\033[102m"
    BG_YELLOW = "\033[103m"
    BG_BLUE = "\033[104m"
    BG_MAGENTA = "\033[105m"
    BG_CYAN = "\033[106m"


def cpu_intensive_task(duration, worker_id, stats_queue):
    """
    Perform CPU-intensive calculations for the specified duration.
    
    Args:
        duration: Time in seconds to run the stress test (0 for continuous)
        worker_id: Identifier for this worker thread
        stats_queue: Queue to send statistics updates
    """
    start_time = time.time()
    end_time = start_time + duration if duration > 0 else float("inf")
    iterations = 0
    flops = 0  # Floating point operations counter
    last_update = start_time
    
    try:
        while time.time() < end_time:
            # Perform CPU-intensive calculations
            for i in range(1000):
                _ = math.sqrt(i) ** 2  # 2 FLOPs (sqrt + power)
                _ = math.factorial(20)  # Not counted as FLOP (integer operations)
                _ = sum([x**2 for x in range(100)])  # 100 FLOPs (power operations)
                flops += 102  # Total FLOPs per inner iteration
            
            iterations += 1000
            
            # Send update every second
            current_time = time.time()
            if current_time - last_update >= 1.0:
                elapsed = current_time - start_time
                flops_per_sec = flops / elapsed
                stats_queue.put({
                    "worker_id": worker_id,
                    "elapsed": elapsed,
                    "flops": flops,
                    "flops_per_sec": flops_per_sec
                })
                last_update = current_time
    except (KeyboardInterrupt, SystemExit):
        # Graceful shutdown
        pass
    
    # Last update
    elapsed = time.time() - start_time
    flops_per_sec = flops / elapsed if elapsed > 0 else 0
    
    return {
        "worker_id": worker_id,
        "iterations": iterations,
        "flops": flops,
        "elapsed": elapsed,
        "flops_per_sec": flops_per_sec
    }


def format_flops(flops, colorize=True):
    """Format FLOPS value with appropriate unit and optional color."""
    if flops >= 1e12:
        value = f"{flops/1e12:.2f} TFLOPS"
        color = Colors.MAGENTA if colorize else ""
    elif flops >= 1e9:
        value = f"{flops/1e9:.2f} GFLOPS"
        color = Colors.CYAN if colorize else ""
    elif flops >= 1e6:
        value = f"{flops/1e6:.2f} MFLOPS"
        color = Colors.GREEN if colorize else ""
    elif flops >= 1e3:
        value = f"{flops/1e3:.2f} KFLOPS"
        color = Colors.YELLOW if colorize else ""
    else:
        value = f"{flops:.2f} FLOPS"
        color = Colors.WHITE if colorize else ""
    
    return f"{color}{value}{Colors.RESET}" if colorize else value


def stats_monitor(stats_queue, cores, duration, show_per_core):
    """Monitor and display real-time statistics from workers."""
    start_time = time.time()
    worker_stats = {i: {"flops": 0, "flops_per_sec": 0} for i in range(1, cores + 1)}
    continuous_mode = duration == 0
    update_count = 0
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Real-time Performance Monitor:{Colors.RESET}")
    if continuous_mode:
        print(f"{Colors.YELLOW}Running in continuous mode - Press Ctrl+C to stop{Colors.RESET}")
    print(f"{Colors.BLUE}{'─' * 80}{Colors.RESET}")
    
    end_time = start_time + duration if duration > 0 else float("inf")
    
    while time.time() < end_time:
        try:
            # Get all available stats updates
            while not stats_queue.empty():
                stat = stats_queue.get_nowait()
                worker_stats[stat["worker_id"]] = {
                    "flops": stat["flops"],
                    "flops_per_sec": stat["flops_per_sec"]
                }
            
            # Calculate totals
            total_flops = sum(w["flops"] for w in worker_stats.values())
            total_flops_per_sec = sum(w["flops_per_sec"] for w in worker_stats.values())
            active_cores = sum(1 for w in worker_stats.values() if w["flops_per_sec"] > 0)
            
            elapsed = time.time() - start_time
            update_count += 1
            
            # Calculate per-core average
            avg_per_core = total_flops_per_sec / cores if cores > 0 else 0
            
            # Display update with more details
            print(f"{Colors.YELLOW}[{elapsed:6.1f}s]{Colors.RESET} "
                  f"{Colors.BOLD}Total:{Colors.RESET} {format_flops(total_flops_per_sec)}/s "
                  f"{Colors.WHITE}|{Colors.RESET} "
                  f"{Colors.BOLD}Per-Core:{Colors.RESET} {format_flops(avg_per_core)}/s "
                  f"{Colors.WHITE}|{Colors.RESET} "
                  f"{Colors.BOLD}Cumulative:{Colors.RESET} {format_flops(total_flops)} "
                  f"{Colors.WHITE}|{Colors.RESET} "
                  f"{Colors.BOLD}Active:{Colors.RESET} {Colors.CYAN}{active_cores}{Colors.RESET}/{Colors.CYAN}{cores}{Colors.RESET}")
            
            # Show per-core breakdown if enabled
            if show_per_core and cores <= 16:
                print(f"{Colors.BLUE}    └─ Core Performance:{Colors.RESET}", end=" ")
                for worker_id in sorted(worker_stats.keys())[:8]:  # Show first 8 cores
                    flops = worker_stats[worker_id]["flops_per_sec"]
                    if flops > 1e6:
                        print(f"{Colors.CYAN}C{worker_id}:{flops/1e6:.0f}M{Colors.RESET}", end=" ")
                if cores > 8:
                    print(f"{Colors.WHITE}...{Colors.RESET}", end="")
                print()
            
            time.sleep(1)
        except:
            pass


def run_stress_test(cores=None, duration=60, show_per_core=False):
    """
    Run the CPU stress test using multiple processes.
    
    Args:
        cores: Number of CPU cores to use (None = all available)
        duration: Duration in seconds to run the test (0 for continuous)
        show_per_core: Show individual core performance stats
    """
    if cores is None:
        cores = multiprocessing.cpu_count()
    else:
        cores = min(cores, multiprocessing.cpu_count())
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}CPU Stress Test with FLOPS Monitoring{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.WHITE}System: {Colors.GREEN}{multiprocessing.cpu_count()}{Colors.WHITE} CPU cores available "
          f"{Colors.WHITE}| Platform: {Colors.CYAN}{sys.platform}{Colors.RESET}")
    print(f"{Colors.WHITE}Using: {Colors.GREEN}{cores}{Colors.WHITE} core(s) for stress test{Colors.RESET}")
    if duration == 0:
        print(f"{Colors.WHITE}Duration: {Colors.YELLOW}Continuous{Colors.WHITE} (press Ctrl+C to stop){Colors.RESET}")
    else:
        print(f"{Colors.WHITE}Duration: {Colors.YELLOW}{duration}{Colors.WHITE} seconds{Colors.RESET}")
    print(f"{Colors.WHITE}Start time: {Colors.MAGENTA}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    
    # Create a queue for statistics
    manager = multiprocessing.Manager()
    stats_queue = manager.Queue()
    
    # Start the statistics monitor in a separate process
    monitor_process = multiprocessing.Process(
        target=stats_monitor,
        args=(stats_queue, cores, duration, show_per_core)
    )
    monitor_process.start()
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=cores) as pool:
        # Start the stress test on all cores
        results = []
        for i in range(cores):
            result = pool.apply_async(cpu_intensive_task, (duration, i+1, stats_queue))
            results.append(result)
        
        # Wait for all workers to complete
        if duration == 0:
            # Continuous mode - wait indefinitely
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                # Gracefully collect results before terminating
                print(f"\n\n{Colors.YELLOW}Stopping stress test...{Colors.RESET}", flush=True)
                
                # Close pool to prevent new tasks, then terminate workers
                pool.close()
                time.sleep(0.5)  # Give workers a moment to finish current iteration
                pool.terminate()
                pool.join()
                
                # Stop monitor
                monitor_process.terminate()
                monitor_process.join()
                
                # Display interruption summary
                print(f"\n{Colors.BOLD}{Colors.YELLOW}{'='*80}{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.YELLOW}⚠ Test Interrupted{Colors.RESET} {Colors.WHITE}at {Colors.MAGENTA}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.YELLOW}{'='*80}{Colors.RESET}")
                print(f"{Colors.WHITE}Test was running in continuous mode.{Colors.RESET}")
                print(f"{Colors.WHITE}Check the real-time monitor output above for performance data.{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.YELLOW}{'='*80}{Colors.RESET}")
                return
        else:
            worker_results = [r.get() for r in results]
    
    # Wait for monitor to finish
    monitor_process.join(timeout=2)
    if monitor_process.is_alive():
        monitor_process.terminate()
    
    # Calculate final statistics (only for timed mode)
    if duration > 0:
        total_iterations = sum(r["iterations"] for r in worker_results)
        total_flops = sum(r["flops"] for r in worker_results)
        avg_flops_per_sec = sum(r["flops_per_sec"] for r in worker_results)
        avg_per_core = avg_flops_per_sec / cores if cores > 0 else 0
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}✓ Test Completed{Colors.RESET} {Colors.WHITE}at {Colors.MAGENTA}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.RESET}")
        print(f"{Colors.WHITE}Duration: {Colors.CYAN}{duration}s{Colors.RESET} "
              f"{Colors.WHITE}| Cores Used: {Colors.CYAN}{cores}{Colors.RESET} "
              f"{Colors.WHITE}| Total Iterations: {Colors.CYAN}{total_iterations:,}{Colors.RESET}")
        print(f"{Colors.WHITE}Total FLOPS: {format_flops(total_flops)}{Colors.RESET}")
        print(f"{Colors.WHITE}Average Performance: {Colors.BOLD}{format_flops(avg_flops_per_sec)}{Colors.WHITE}/s{Colors.RESET} "
              f"{Colors.WHITE}({format_flops(avg_per_core)}/s per core){Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.RESET}")


def run_gpu_stress_test(duration=60, show_verbose=False):
    """
    Run GPU stress test using CUDA operations.
    
    Args:
        duration: Duration in seconds to run the test (0 for continuous)
        show_verbose: Show detailed GPU information
    """
    if GPU_LIBRARY == "pytorch":
        import torch
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif GPU_LIBRARY == "cupy":
        import cupy as cp
        device = cp.cuda.Device(0)
        gpu_name = device.name
        gpu_memory = device.mem_info[1] / (1024**3)
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}GPU Stress Test with TFLOPS Monitoring{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.WHITE}GPU: {Colors.GREEN}{gpu_name}{Colors.RESET}")
    print(f"{Colors.WHITE}Memory: {Colors.GREEN}{gpu_memory:.2f} GB{Colors.RESET}")
    print(f"{Colors.WHITE}Library: {Colors.CYAN}{GPU_LIBRARY.upper()}{Colors.RESET}")
    if duration == 0:
        print(f"{Colors.WHITE}Duration: {Colors.YELLOW}Continuous{Colors.WHITE} (press Ctrl+C to stop){Colors.RESET}")
    else:
        print(f"{Colors.WHITE}Duration: {Colors.YELLOW}{duration}{Colors.WHITE} seconds{Colors.RESET}")
    print(f"{Colors.WHITE}Start time: {Colors.MAGENTA}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Real-time Performance Monitor:{Colors.RESET}")
    if duration == 0:
        print(f"{Colors.YELLOW}Running in continuous mode - Press Ctrl+C to stop{Colors.RESET}")
    print(f"{Colors.BLUE}{'─' * 80}{Colors.RESET}")
    
    start_time = time.time()
    end_time = start_time + duration if duration > 0 else float('inf')
    total_flops = 0
    iteration = 0
    
    try:
        if GPU_LIBRARY == "pytorch":
            # Use PyTorch for GPU calculations
            matrix_size = 4096
            
            while time.time() < end_time:
                iteration += 1
                iter_start = time.time()
                
                # Matrix multiplication: 2*N^3 FLOPS
                a = torch.randn(matrix_size, matrix_size, device=device)
                b = torch.randn(matrix_size, matrix_size, device=device)
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                iter_time = time.time() - iter_start
                flops_this_iter = 2 * (matrix_size ** 3)
                total_flops += flops_this_iter
                
                elapsed = time.time() - start_time
                tflops_per_sec = (total_flops / elapsed) / 1e12
                
                if iteration % 10 == 0:
                    print(f"{Colors.YELLOW}[{elapsed:6.1f}s]{Colors.RESET} "
                          f"{Colors.BOLD}Performance:{Colors.RESET} {Colors.MAGENTA}{tflops_per_sec:.2f} TFLOPS{Colors.RESET} "
                          f"{Colors.WHITE}|{Colors.RESET} "
                          f"{Colors.BOLD}Iterations:{Colors.RESET} {Colors.CYAN}{iteration}{Colors.RESET} "
                          f"{Colors.WHITE}|{Colors.RESET} "
                          f"{Colors.BOLD}Cumulative:{Colors.RESET} {Colors.MAGENTA}{total_flops/1e12:.2f} TFLOPS{Colors.RESET}")
        
        elif GPU_LIBRARY == "cupy":
            # Use CuPy for GPU calculations
            matrix_size = 4096
            
            while time.time() < end_time:
                iteration += 1
                iter_start = time.time()
                
                a = cp.random.randn(matrix_size, matrix_size)
                b = cp.random.randn(matrix_size, matrix_size)
                c = cp.matmul(a, b)
                cp.cuda.Stream.null.synchronize()
                
                iter_time = time.time() - iter_start
                flops_this_iter = 2 * (matrix_size ** 3)
                total_flops += flops_this_iter
                
                elapsed = time.time() - start_time
                tflops_per_sec = (total_flops / elapsed) / 1e12
                
                if iteration % 10 == 0:
                    print(f"{Colors.YELLOW}[{elapsed:6.1f}s]{Colors.RESET} "
                          f"{Colors.BOLD}Performance:{Colors.RESET} {Colors.MAGENTA}{tflops_per_sec:.2f} TFLOPS{Colors.RESET} "
                          f"{Colors.WHITE}|{Colors.RESET} "
                          f"{Colors.BOLD}Iterations:{Colors.RESET} {Colors.CYAN}{iteration}{Colors.RESET} "
                          f"{Colors.WHITE}|{Colors.RESET} "
                          f"{Colors.BOLD}Cumulative:{Colors.RESET} {Colors.MAGENTA}{total_flops/1e12:.2f} TFLOPS{Colors.RESET}")
    
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Stopping GPU stress test...{Colors.RESET}", flush=True)
        elapsed = time.time() - start_time
        
        print(f"\n{Colors.BOLD}{Colors.YELLOW}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.YELLOW}⚠ Test Interrupted{Colors.RESET} {Colors.WHITE}at {Colors.MAGENTA}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.YELLOW}{'='*80}{Colors.RESET}")
        print(f"{Colors.WHITE}Test was running in continuous mode.{Colors.RESET}")
        print(f"{Colors.WHITE}Check the real-time monitor output above for performance data.{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.YELLOW}{'='*80}{Colors.RESET}")
        return
    
    # Final summary for timed tests
    if duration > 0:
        elapsed = time.time() - start_time
        avg_tflops = (total_flops / elapsed) / 1e12
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}✓ Test Completed{Colors.RESET} {Colors.WHITE}at {Colors.MAGENTA}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.RESET}")
        print(f"{Colors.WHITE}Duration: {Colors.CYAN}{duration}s{Colors.RESET} "
              f"{Colors.WHITE}| Iterations: {Colors.CYAN}{iteration}{Colors.RESET}")
        print(f"{Colors.WHITE}Total FLOPS: {Colors.MAGENTA}{total_flops/1e12:.2f} TFLOPS{Colors.RESET}")
        print(f"{Colors.WHITE}Average Performance: {Colors.BOLD}{Colors.MAGENTA}{avg_tflops:.2f} TFLOPS/s{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.RESET}")


def main():
    """Main entry point for the CPU/GPU stress tester."""
    parser = argparse.ArgumentParser(
        description="CPU/GPU Stress Tester - Test your CPU or GPU with intensive calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Use all CPU cores for 60 seconds
  %(prog)s -c 4 -d 30              # Use 4 CPU cores for 30 seconds
  %(prog)s --cores 2 --duration 120 # Use 2 CPU cores for 120 seconds
  %(prog)s -d 0                     # Run continuously until Ctrl+C
  %(prog)s -v -d 30                 # Show per-core breakdown (verbose mode)
  %(prog)s --mode gpu -d 60         # Test GPU for 60 seconds
  %(prog)s --mode gpu -d 0          # Test GPU continuously
        """
    )
    
    parser.add_argument(
        "-c", "--cores",
        type=int,
        default=None,
        help="Number of CPU cores to use (default: all available cores)"
    )
    
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=60,
        help="Duration in seconds to run the test (default: 60, use 0 for continuous)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show per-core performance breakdown (CPU mode only)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Test mode: 'cpu' for CPU testing (default) or 'gpu' for GPU testing"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.cores is not None and args.cores < 1:
        print("Error: Number of cores must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    if args.duration < 0:
        print("Error: Duration cannot be negative", file=sys.stderr)
        sys.exit(1)
    
    # Check GPU mode requirements
    if args.mode == "gpu":
        if not GPU_AVAILABLE:
            print(f"{Colors.RED}Error: GPU mode requires PyTorch with CUDA or CuPy.{Colors.RESET}", file=sys.stderr)
            print(f"{Colors.YELLOW}Install with: pip install torch (with CUDA) or pip install cupy{Colors.RESET}", file=sys.stderr)
            sys.exit(1)
    
    try:
        if args.mode == "gpu":
            run_gpu_stress_test(duration=args.duration, show_verbose=args.verbose)
        else:
            run_stress_test(cores=args.cores, duration=args.duration, show_per_core=args.verbose)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Stress test interrupted by user{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Error during stress test: {e}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()