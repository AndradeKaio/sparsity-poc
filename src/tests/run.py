import subprocess
import re
import sys

def parse_config_file(file_path):
    configs = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            match = re.match(r"\(\((\d+), (\d+)\), (\w+), ([0-9.]+), (\w+), (\w+)\)", line)
            if match:
                size = (int(match.group(1)), int(match.group(2)))
                format_type = match.group(3)
                sparsity = float(match.group(4))
                input_conv = match.group(5).lower() == "true"
                sampling = match.group(6)
                configs.append((size, format_type, sparsity, input_conv, sampling))
    return configs

def run_binary(binary_path, args, num_runs):
    total_time = 0.0
    for _ in range(num_runs):
        result = subprocess.run([binary_path] + list(map(str, args)), capture_output=True, text=True)
        output_lines = result.stdout.strip().split("\n")
        time_line = output_lines[-1]
        try:
            exec_time = float(time_line)
            total_time += exec_time
        except ValueError:
            print(f"Warning: Could not parse execution time from output: {time_line}")
    return total_time / num_runs

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <config_file> <num_runs> <binary_path>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    num_runs = int(sys.argv[2])
    binary_path = sys.argv[3]
    output_file = "out.txt"
    
    configs = parse_config_file(config_file)
    with open(output_file, "wt") as output:
        for config in configs:
            args = [config[0][0], config[0][1], config[1], config[2], config[3], config[4]]
            avg_time = run_binary(binary_path, args, num_runs)
            output.write(f"{args}:{avg_time:.2f}ms\n")
            print(f"Executed {args}: {avg_time:.2f} ms")

if __name__ == "__main__":
    main()
import subpr
