#!/usr/bin/env python3
"""
Visualize response time trends on a timeline (x-axis = actual time when question was sent)
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

def visualize_response_time_timeline(csv_path: str, output_path: str = None):
    """
    Create a single plot showing response time trends over actual timeline

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

    # Get unique users
    users = sorted(df_success['user_id'].unique())

    # Create figure
    plt.figure(figsize=(14, 8))

    # Color palette for 10 users
    colors = plt.cm.tab10(range(10))

    # Plot each user's response time trend
    for idx, user_id in enumerate(users):
        user_data = df_success[df_success['user_id'] == user_id].sort_values('relative_time_sec')

        plt.plot(
            user_data['relative_time_sec'] / 60,  # Convert to minutes
            user_data['response_time_sec'],
            marker='o',
            markersize=3,
            linewidth=1.5,
            alpha=0.7,
            color=colors[idx],
            label=f'User {user_id}'
        )

    # Formatting
    plt.xlabel('Time Since Test Start (minutes)', fontsize=12, fontweight='bold')
    plt.ylabel('Response Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Response Time Over Timeline (10 Users, Continuous Flow)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', ncol=2, fontsize=9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(0, df_success['relative_time_sec'].max() / 60 * 1.05)
    plt.ylim(0, df_success['response_time_sec'].max() * 1.1)

    # Add horizontal line for average
    avg_response_time = df_success['response_time_sec'].mean()
    plt.axhline(y=avg_response_time, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Overall Avg: {avg_response_time:.2f}s')

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
    else:
        plt.show()

    # Print timeline statistics
    print("\n" + "=" * 60)
    print("Timeline Statistics")
    print("=" * 60)
    test_duration = df_success['relative_time_sec'].max() / 60
    print(f"Test Start: {test_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Duration: {test_duration:.2f} minutes")
    print(f"Total Requests: {len(df_success)}")
    print(f"Avg Requests/minute: {len(df_success) / test_duration:.1f}")

    print("\n" + "=" * 60)
    print("Response Time Statistics by User")
    print("=" * 60)
    for user_id in users:
        user_data = df_success[df_success['user_id'] == user_id].sort_values('relative_time_sec')
        start_time = user_data['relative_time_sec'].min() / 60
        end_time = user_data['relative_time_sec'].max() / 60
        duration = end_time - start_time

        print(f"User {user_id}:")
        print(f"  - Start: {start_time:.2f} min")
        print(f"  - End: {end_time:.2f} min")
        print(f"  - Duration: {duration:.2f} min")
        print(f"  - Avg Response: {user_data['response_time_sec'].mean():.2f}s")

    print("\n" + "=" * 60)
    print(f"Overall Average Response Time: {avg_response_time:.2f}s")
    print("=" * 60)


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
    output_path = os.path.join(results_dir, f"response_time_timeline_{timestamp}.png")

    print(f"📊 Visualizing: {csv_path}")
    visualize_response_time_timeline(csv_path, output_path)
