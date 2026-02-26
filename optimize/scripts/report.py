#!/usr/bin/env python3
import os
import pandas as pd
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def main():
    progress_csv = "optimize/logs/progress.csv"
    report_md = "optimize/logs/optimization_report.md"

    if not os.path.exists(progress_csv):
        print(f"Error: {progress_csv} not found.")
        return

    df = pd.read_csv(progress_csv)

    with open(report_md, 'w') as f:
        f.write("# Time-Stretch Optimization Report\n\n")
        f.write("## Score Progression\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")

        if plt and not df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(df['iteration'], df['avg_score'], label='Average Score', marker='o')
            plt.plot(df['iteration'], df['worst_score'], label='Worst Score', marker='x')
            plt.axhline(y=92.0, color='r', linestyle='--', label='Target')
            plt.xlabel('Iteration')
            plt.ylabel('Score')
            plt.title('Optimization Progress')
            plt.legend()
            plt.grid(True)

            plot_path = "optimize/logs/progress_chart.png"
            plt.savefig(plot_path)
            f.write("![Optimization Progress](progress_chart.png)\n\n")

        f.write("## Summary\n")
        if not df.empty:
            best_iter = df.loc[df['avg_score'].idxmax()]
            f.write(f"- **Best Average Score:** {best_iter['avg_score']:.2f} (Iteration {best_iter['iteration']})\n")
            f.write(f"- **Total Iterations:** {len(df)}\n")
            f.write(f"- **Initial Score:** {df.iloc[0]['avg_score']:.2f}\n")
            f.write(f"- **Final Score:** {df.iloc[-1]['avg_score']:.2f}\n")

        f.write("\n## Git History\n")
        for idx, row in df.iterrows():
            f.write(f"- Iteration {row['iteration']}: {row['git_sha']} (Score: {row['avg_score']:.2f})\n")

    print(f"Report generated at {report_md}")

if __name__ == "__main__":
    main()
