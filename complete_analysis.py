#!/usr/bin/env python3

import os
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

# Configuration - adjust these paths as needed
RESULTS_DIR = "/home/vasilis/Downloads/AeromineRunner/results"
SECTIONS_CSV = f"{RESULTS_DIR}/sections.csv"
SUMMARY_CSV = f"{RESULTS_DIR}/sections_summary.csv"
OUTPUT_DIR = f"{RESULTS_DIR}/complete_analysis"
POINT_SIZE = 1

def create_output_dir():
    """Create output directory if it doesn't exist"""
    out_dir = pathlib.Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)
    return out_dir

def check_files():
    """Check if required files exist"""
    if not pathlib.Path(SECTIONS_CSV).exists():
        raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο: {SECTIONS_CSV}")

    if not pathlib.Path(SUMMARY_CSV).exists():
        raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο: {SUMMARY_CSV}")

def load_data():
    """Load both CSV files"""
    print(f"Loading data from: {RESULTS_DIR}")

    # Load sections data for plotting
    df_sections = pd.read_csv(SECTIONS_CSV)
    required_sections = ["dist_off", "z", "section_id"]
    missing_sections = [c for c in required_sections if c not in df_sections.columns]
    if missing_sections:
        raise ValueError(f"Sections CSV must contain columns: {required_sections}. Found: {df_sections.columns.tolist()}")

    # Load summary data for analysis
    df_summary = pd.read_csv(SUMMARY_CSV)
    required_summary = {"section_id", "wall_distance", "depth"}
    missing_summary = [c for c in required_summary if c not in df_summary.columns]
    if missing_summary:
        raise ValueError(f"Summary CSV must contain columns: {required_summary}. Found: {df_summary.columns.tolist()}")

    print(f"Loaded {len(df_sections):,} points from sections.csv")
    print(f"Loaded {len(df_summary)} sections from sections_summary.csv")

    return df_sections, df_summary

def create_cross_section_plots(df_sections, out_dir):
    """Create individual cross-section plots (from test.py)"""
    print("\n📊 Creating cross-section plots...")

    section_ids = sorted(df_sections["section_id"].unique())
    print(f"Found {len(section_ids)} sections: {section_ids[0]} to {section_ids[-1]}")

    for sid in section_ids:
        sub = df_sections[df_sections["section_id"] == sid]

        plt.figure(figsize=(12, 6))
        plt.scatter(sub["dist_off"], sub["z"], s=POINT_SIZE)
        plt.xlabel("dist_off (m)")
        plt.ylabel("z (m)")
        plt.title(f"Trench cross-section — section {sid}")
        plt.grid(True)
        plt.tight_layout()

        fname = f"section_{int(sid):05d}.jpg"
        out_path = out_dir / fname
        plt.savefig(out_path, dpi=150)
        plt.close()

    print(f"✅ Saved {len(section_ids)} cross-section plots")

def create_summary_analysis_plots(df_summary, out_dir):
    """Create summary analysis plots (from analyze_summary.py)"""
    print("\n📈 Creating summary analysis plots...")

    # Sort by section_id for proper plotting
    df = df_summary.sort_values("section_id")

    # Plot 1: Wall Distance vs Section ID
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df["section_id"], df["wall_distance"], 'b-', linewidth=2, marker='o', markersize=3)
    plt.xlabel("Section ID")
    plt.ylabel("Wall Distance (m)")
    plt.title("Trench Wall Distance Along Length")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=df["wall_distance"].mean(), color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {df["wall_distance"].mean():.3f}m')
    plt.legend()

    # Plot 2: Depth vs Section ID
    plt.subplot(2, 1, 2)
    plt.plot(df["section_id"], df["depth"], 'g-', linewidth=2, marker='s', markersize=3)
    plt.xlabel("Section ID")
    plt.ylabel("Depth (m)")
    plt.title("Trench Depth Along Length")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=df["depth"].mean(), color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {df["depth"].mean():.3f}m')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "trench_profile_analysis.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Plot 3: Combined Wall Distance and Depth
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Wall distance on primary y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Section ID')
    ax1.set_ylabel('Wall Distance (m)', color=color1)
    line1 = ax1.plot(df["section_id"], df["wall_distance"], 'b-', linewidth=2, marker='o', markersize=4, label='Wall Distance')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Depth on secondary y-axis
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Depth (m)', color=color2)
    line2 = ax2.plot(df["section_id"], df["depth"], 'g-', linewidth=2, marker='s', markersize=4, label='Depth')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title("Trench Profile: Wall Distance & Depth Analysis")
    plt.tight_layout()
    plt.savefig(out_dir / "combined_profile_analysis.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Plot 4: Statistics Summary
    plt.figure(figsize=(12, 8))

    # Wall Distance Stats
    plt.subplot(2, 2, 1)
    plt.hist(df["wall_distance"], bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(df["wall_distance"].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df["wall_distance"].mean():.3f}m')
    plt.xlabel("Wall Distance (m)")
    plt.ylabel("Frequency")
    plt.title("Wall Distance Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Depth Stats
    plt.subplot(2, 2, 2)
    plt.hist(df["depth"], bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(df["depth"].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df["depth"].mean():.3f}m')
    plt.xlabel("Depth (m)")
    plt.ylabel("Frequency")
    plt.title("Depth Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Wall Distance vs Depth scatter
    plt.subplot(2, 2, 3)
    plt.scatter(df["wall_distance"], df["depth"], alpha=0.6, s=30, color='purple')
    plt.xlabel("Wall Distance (m)")
    plt.ylabel("Depth (m)")
    plt.title("Wall Distance vs Depth Correlation")
    plt.grid(True, alpha=0.3)

    # Point count per section
    plt.subplot(2, 2, 4)
    plt.bar(df["section_id"], df["count"], alpha=0.7, color='orange', width=0.8)
    plt.xlabel("Section ID")
    plt.ylabel("Point Count")
    plt.title("Points per Section")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "statistics_summary.png", dpi=200, bbox_inches='tight')
    plt.close()

    print("✅ Saved summary analysis plots")

def generate_report(df_summary, out_dir):
    """Generate detailed text report"""
    print("\n📄 Generating analysis report...")

    df = df_summary.sort_values("section_id")

    # Print summary statistics
    print("\n📊 SUMMARY STATISTICS:")
    print(f"Total sections: {len(df)}")
    print(f"Section ID range: {df['section_id'].min()} to {df['section_id'].max()}")

    print("\n🏗️ WALL DISTANCE:")
    print(f"  Mean: {df['wall_distance'].mean():.3f}m")
    print(f"  Min:  {df['wall_distance'].min():.3f}m")
    print(f"  Max:  {df['wall_distance'].max():.3f}m")
    print(f"  Std:  {df['wall_distance'].std():.3f}m")

    print("\n📏 DEPTH:")
    print(f"  Mean: {df['depth'].mean():.3f}m")
    print(f"  Min:  {df['depth'].min():.3f}m")
    print(f"  Max:  {df['depth'].max():.3f}m")
    print(f"  Std:  {df['depth'].std():.3f}m")

    print("\n📈 VARIABILITY:")
    wall_cv = df['wall_distance'].std() / df['wall_distance'].mean() * 100
    depth_cv = df['depth'].std() / df['depth'].mean() * 100
    print(f"  Wall distance CV: {wall_cv:.1f}%")
    print(f"  Depth CV: {depth_cv:.1f}%")

    # Save detailed report to file
    report_file = out_dir / "complete_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("📊 COMPLETE TRENCH ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Directory: {RESULTS_DIR}\n")
        f.write(f"Sections File: {SECTIONS_CSV}\n")
        f.write(f"Summary File: {SUMMARY_CSV}\n\n")

        f.write("📊 SUMMARY STATISTICS:\n")
        f.write(f"Total sections: {len(df)}\n")
        f.write(f"Section ID range: {df['section_id'].min()} to {df['section_id'].max()}\n\n")

        f.write("🏗️ WALL DISTANCE:\n")
        f.write(f"  Mean: {df['wall_distance'].mean():.3f}m\n")
        f.write(f"  Min:  {df['wall_distance'].min():.3f}m\n")
        f.write(f"  Max:  {df['wall_distance'].max():.3f}m\n")
        f.write(f"  Std:  {df['wall_distance'].std():.3f}m\n\n")

        f.write("📏 DEPTH:\n")
        f.write(f"  Mean: {df['depth'].mean():.3f}m\n")
        f.write(f"  Min:  {df['depth'].min():.3f}m\n")
        f.write(f"  Max:  {df['depth'].max():.3f}m\n")
        f.write(f"  Std:  {df['depth'].std():.3f}m\n\n")

        f.write("📈 VARIABILITY:\n")
        f.write(f"  Wall distance CV: {wall_cv:.1f}%\n")
        f.write(f"  Depth CV: {depth_cv:.1f}%\n\n")

        f.write("📋 DETAILED SECTION DATA:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Section':<8} {'Wall Dist':<10} {'Depth':<8} {'Points':<8} {'Z_Range':<15}\n")
        f.write("-" * 90 + "\n")
        for _, row in df.iterrows():
            z_range = f"{row['z_min']:.2f}-{row['z_max']:.2f}"
            f.write(f"{int(row['section_id']):<8} {row['wall_distance']:<10.3f} {row['depth']:<8.3f} {int(row['count']):<8} {z_range:<15}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Generated by AeromineRunner complete_analysis.py\n")
        f.write(f"Output directory: {out_dir.resolve()}\n")

    print(f"📄 Detailed report saved to: {report_file}")

def main():
    """Main function combining test.py and analyze_summary.py"""
    try:
        print("🚀 Starting Complete Trench Analysis")
        print("=" * 50)

        # Setup
        out_dir = create_output_dir()
        check_files()

        # Load data
        df_sections, df_summary = load_data()

        # Generate all outputs
        create_cross_section_plots(df_sections, out_dir)
        create_summary_analysis_plots(df_summary, out_dir)
        generate_report(df_summary, out_dir)

        print("\n✅ COMPLETE ANALYSIS FINISHED!")
        print(f"📁 All results saved in: {out_dir.resolve()}")
        print("\n📂 Generated files:")
        print("  - Individual cross-section plots (section_XXXXX.jpg)")
        print("  - trench_profile_analysis.png (wall distance & depth profiles)")
        print("  - combined_profile_analysis.png (dual-axis combined view)")
        print("  - statistics_summary.png (histograms & correlations)")
        print("  - complete_analysis_report.txt (detailed text report)")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())