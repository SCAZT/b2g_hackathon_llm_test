#!/usr/bin/env python3
"""
Visualize response time trends across 60 questions for all users
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def visualize_response_time(csv_path: str, output_path: str = None):
    """
    Create a single plot showing response time trends for all users

    Args:
        csv_path: Path to stress_test_details CSV file
        output_path: Path to save the plot (optional)
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Filter only successful requests
    df_success = df[df['status'] == 'success'].copy()

    # Get unique users
    users = sorted(df_success['user_id'].unique())

    # Create figure
    plt.figure(figsize=(14, 8))

    # Color palette for 10 users
    colors = plt.cm.tab10(range(10))

    # Plot each user's response time trend
    for idx, user_id in enumerate(users):
        user_data = df_success[df_success['user_id'] == user_id].sort_values('question_num')

        plt.plot(
            user_data['question_num'],
            user_data['response_time_sec'],
            marker='o',
            markersize=3,
            linewidth=1.5,
            alpha=0.7,
            color=colors[idx],
            label=f'User {user_id}'
        )

    # Formatting
    plt.xlabel('Question Number', fontsize=12, fontweight='bold')
    plt.ylabel('Response Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Response Time Trends Across 60 Questions (10 Users)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', ncol=2, fontsize=9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(0, 61)
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

    # Print statistics
    print("\n" + "=" * 60)
    print("Response Time Statistics by User")
    print("=" * 60)
    for user_id in users:
        user_data = df_success[df_success['user_id'] == user_id]
        print(f"User {user_id}:")
        print(f"  - Min: {user_data['response_time_sec'].min():.2f}s")
        print(f"  - Max: {user_data['response_time_sec'].max():.2f}s")
        print(f"  - Avg: {user_data['response_time_sec'].mean():.2f}s")
        print(f"  - Std: {user_data['response_time_sec'].std():.2f}s")

    print("\n" + "=" * 60)
    print(f"Overall Average: {avg_response_time:.2f}s")
    print(f"Overall Std: {df_success['response_time_sec'].std():.2f}s")
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
    output_path = os.path.join(results_dir, f"response_time_plot_{timestamp}.png")

    print(f"📊 Visualizing: {csv_path}")
    visualize_response_time(csv_path, output_path)
