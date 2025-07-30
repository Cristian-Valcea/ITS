#!/usr/bin/env python3
"""
Performance regression check script for GitHub Actions
"""
import re
import sys

def main():
    try:
        with open('current_perf.txt', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print('No performance data found')
        return

    # Extract P50 latencies
    p50_matches = re.findall(r'P50: ([\d.]+)µs', content)
    if p50_matches:
        max_p50 = max(float(x) for x in p50_matches)
        print(f'Maximum P50 latency: {max_p50}µs')
        
        # Fail if any P50 > 1000µs (1ms)
        if max_p50 > 1000.0:
            print(f'PERFORMANCE REGRESSION: P50 latency {max_p50}µs exceeds 1000µs threshold')
            sys.exit(1)
        else:
            print('Performance within acceptable limits')
    else:
        print('No latency data found')

if __name__ == '__main__':
    main()