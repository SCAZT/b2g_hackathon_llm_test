#!/usr/bin/env python3
"""
Visualize question sending timeline as event plot
Each user has a horizontal timeline showing when questions were sent
The gap between adjacent points = response time for that question
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
from datetime import datetime
import numpy as np

def visualize_timeline_eventplot(csv_path: str, output_path: str = None):
    """
    Create an event plot showing question send times for each user

    Args:
        csv_path: Path to stress_test_details CSV file
        output_path: Path to save the plot (optional)
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Filter only successful requests
    df_success = df[df['status'] == 'success'].copy()

    # Convert sent_time to datetime
    df_success['sent_datetime'] = pd.to_datetime(df_success['sent_time'])

    # Calculate relative time (seconds from test start)
    test_start = df_success['sent_datetime'].min()
    df_success['relative_time_sec'] = (df_success['sent_datetime'] - test_start).dt.total_seconds()
    df_success['relative_time_min'] = df_success['relative_time_sec'] / 60

    # Get unique users
    users = sorted(df_success['user_id'].unique())

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Color map for response time (blue=fast, red=slow)
    response_times = df_success['response_time_sec'].values
    vmin = response_times.min()
    vmax = response_times.max()

    # Plot each user's timeline
    for idx, user_id in enumerate(users):
        user_data = df_success[df_success['user_id'] == user_id].sort_values('relative_time_min')

        # Use scatter plot to allow color coding by response time
        scatter = ax.scatter(
            user_data['relative_time_min'],
            [idx] * len(user_data),  # Y position = user index
            c=user_data['response_time_sec'],  # Color by response time
            cmap='RdYlGn_r',  # Red=slow, Green=fast
            vmin=vmin,
            vmax=vmax,
            s=50,  # Marker size
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5,
            zorder=3
        )

        # Draw connecting lines to show sequence
        ax.plot(
            user_data['relative_time_min'],
            [idx] * len(user_data),
            color='gray',
            linewidth=1,
            alpha=0.3,
            zorder=1
        )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Response Time (seconds)', fontsize=11, fontweight='bold')

    # Formatting
    ax.set_xlabel('Time Since Test Start (minutes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('User ID', fontsize=12, fontweight='bold')
    ax.set_title('Question Send Timeline (Gap Between Points = Response Time)',
                 fontsize=14, fontweight='bold', pad=20)

    # Set Y-axis labels to user IDs
    ax.set_yticks(range(len(users)))
    ax.set_yticklabels([f'User {uid}' for uid in users])

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_axisbelow(True)

    # Set limits
    ax.set_xlim(-0.5, df_success['relative_time_min'].max() * 1.05)
    ax.set_ylim(-0.5, len(users) - 0.5)

    # Add vertical lines for minute markers
    max_time = df_success['relative_time_min'].max()
    for minute in range(0, int(max_time) + 2):
        ax.axvline(x=minute, color='lightgray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
    else:
        plt.show()

    # Print statistics
    print("\n" + "=" * 70)
    print("Timeline Event Analysis")
    print("=" * 70)

    test_duration = df_success['relative_time_min'].max()
    print(f"Test Start: {test_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Duration: {test_duration:.2f} minutes")
    print(f"Total Requests: {len(df_success)}")

    print("\n" + "=" * 70)
    print("Per-User Timeline Statistics")
    print("=" * 70)
    print(f"{'User':<12} {'Questions':<12} {'Start (min)':<15} {'End (min)':<15} {'Duration (min)':<15}")
    print("-" * 70)

    for user_id in users:
        user_data = df_success[df_success['user_id'] == user_id].sort_values('relative_time_min')
        start_time = user_data['relative_time_min'].min()
        end_time = user_data['relative_time_min'].max()
        duration = end_time - start_time
        count = len(user_data)

        print(f"{user_id:<12} {count:<12} {start_time:<15.2f} {end_time:<15.2f} {duration:<15.2f}")

    # Response time distribution
    print("\n" + "=" * 70)
    print("Response Time Distribution")
    print("=" * 70)
    print(f"Min Response Time: {vmin:.2f}s (fastest)")
    print(f"Max Response Time: {vmax:.2f}s (slowest)")
    print(f"Avg Response Time: {response_times.mean():.2f}s")
    print(f"Std Response Time: {response_times.std():.2f}s")

    # Identify bottleneck periods (high concurrent load)
    print("\n" + "=" * 70)
    print("Concurrency Analysis (requests active at each minute)")
    print("=" * 70)

    for minute in range(0, int(test_duration) + 1):
        # Count how many requests were sent in this minute
        sent_in_minute = df_success[
            (df_success['relative_time_min'] >= minute) &
            (df_success['relative_time_min'] < minute + 1)
        ]
        print(f"Minute {minute}-{minute+1}: {len(sent_in_minute)} questions sent")

    print("=" * 70)


if __name__ == "__main__":
    # Default to latest test results
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, "results")

    # Find latest CSV file
    csv_files = [f for f in os.listdir(results_dir) if f.startswith("stress_test_details_") and f.endswith(".csv")]

    if not csv_files:
        print("❌ No test result CSV files found in results/")
        sys.exit(1)

    # Use latest file
    latest_csv = sorted(csv_files)[-1]
    csv_path = os.path.join(results_dir, latest_csv)

    # Generate output filename
    timestamp = latest_csv.replace("stress_test_details_", "").replace(".csv", "")
    output_path = os.path.join(results_dir, f"timeline_eventplot_{timestamp}.png")

    print(f"📊 Visualizing: {csv_path}")
    visualize_timeline_eventplot(csv_path, output_path)
