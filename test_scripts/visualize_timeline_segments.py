#!/usr/bin/env python3
"""
Visualize question response times as segmented timeline bars
Each user has a horizontal bar divided into segments (one per question)
Segment width = response time
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import sys
import os
from datetime import datetime
import numpy as np

def visualize_timeline_segments(csv_path: str, output_path: str = None):
    """
    Create a segmented timeline visualization where each segment represents
    the response time of one question

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
    fig, ax = plt.subplots(figsize=(18, 10))

    # Color map for response time (green=fast, red=slow)
    response_times = df_success['response_time_sec'].values
    vmin = response_times.min()
    vmax = response_times.max()
    cmap = plt.cm.RdYlGn_r  # Red=slow, Green=fast

    # Normalize for color mapping
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Bar height
    bar_height = 0.6

    # Plot each user's segmented timeline
    for idx, user_id in enumerate(users):
        user_data = df_success[df_success['user_id'] == user_id].sort_values('relative_time_min')

        # For each question, draw a rectangle
        for _, row in user_data.iterrows():
            # Rectangle position
            x_start = row['relative_time_min']
            y_position = idx - bar_height / 2
            width = row['response_time_sec'] / 60  # Convert to minutes
            height = bar_height

            # Color based on response time
            color = cmap(norm(row['response_time_sec']))

            # Draw rectangle
            rect = mpatches.Rectangle(
                (x_start, y_position),
                width,
                height,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.9
            )
            ax.add_patch(rect)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Response Time (seconds)', fontsize=11, fontweight='bold')

    # Formatting
    ax.set_xlabel('Time Since Test Start (minutes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('User ID', fontsize=12, fontweight='bold')
    ax.set_title('Segmented Response Timeline (Each Segment = One Question\'s Response Time)',
                 fontsize=14, fontweight='bold', pad=20)

    # Set Y-axis labels to user IDs
    ax.set_yticks(range(len(users)))
    ax.set_yticklabels([f'User {uid}' for uid in users])

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_axisbelow(True)

    # Set limits
    max_time = df_success['relative_time_min'].max() + df_success['response_time_sec'].max() / 60
    ax.set_xlim(-0.5, max_time * 1.05)
    ax.set_ylim(-0.5, len(users) - 0.5)

    # Add vertical lines for minute markers
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
    print("Segmented Timeline Analysis")
    print("=" * 70)

    test_duration = max_time
    print(f"Test Start: {test_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Duration (including last response): {test_duration:.2f} minutes")
    print(f"Total Requests: {len(df_success)}")

    print("\n" + "=" * 70)
    print("Per-User Timeline Statistics")
    print("=" * 70)
    print(f"{'User':<12} {'Questions':<12} {'Total Time (min)':<20} {'Avg Response (s)':<20}")
    print("-" * 70)

    for user_id in users:
        user_data = df_success[df_success['user_id'] == user_id].sort_values('relative_time_min')
        count = len(user_data)

        # Total time = sum of all response times
        total_time_min = user_data['response_time_sec'].sum() / 60
        avg_response = user_data['response_time_sec'].mean()

        print(f"{user_id:<12} {count:<12} {total_time_min:<20.2f} {avg_response:<20.2f}")

    # Response time distribution
    print("\n" + "=" * 70)
    print("Response Time Distribution")
    print("=" * 70)
    print(f"Min Response Time: {vmin:.2f}s (fastest)")
    print(f"Max Response Time: {vmax:.2f}s (slowest)")
    print(f"Avg Response Time: {response_times.mean():.2f}s")
    print(f"Std Response Time: {response_times.std():.2f}s")

    # Identify slowest responses
    print("\n" + "=" * 70)
    print("Top 5 Slowest Responses")
    print("=" * 70)
    slowest = df_success.nlargest(5, 'response_time_sec')[['user_id', 'question_num', 'response_time_sec', 'relative_time_min']]
    print(f"{'User':<12} {'Question #':<15} {'Response (s)':<15} {'Time (min)':<15}")
    print("-" * 70)
    for _, row in slowest.iterrows():
        print(f"{row['user_id']:<12} {row['question_num']:<15} {row['response_time_sec']:<15.2f} {row['relative_time_min']:<15.2f}")

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
    output_path = os.path.join(results_dir, f"timeline_segments_{timestamp}.png")

    print(f"📊 Visualizing: {csv_path}")
    visualize_timeline_segments(csv_path, output_path)
