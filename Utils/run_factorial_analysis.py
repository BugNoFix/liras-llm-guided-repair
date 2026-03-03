import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# --- This is the part I need to add to handle the "no file found" error ---
# I will assume the file is in the current directory if it's not found in the default path
# This is a common issue when running scripts from a different location.

def get_data_path(args):
    """
    Heuristic to find the data file.
    1. Check the path provided in args.input
    2. Check the current directory for 'combined_run_histories.csv'
    3. Check the parent directory for 'combined_run_histories.csv' (common in project structures)
    4. Check 'Report/Histories/' relative to the current directory.
    """
    potential_paths = [
        args.input,
        'combined_run_histories.csv',
        '../combined_run_histories.csv',
        'Report/Histories/combined_run_histories.csv'
    ]
    for path in potential_paths:
        if os.path.exists(path):
            print(f"Found data file at: {path}")
            return path
    
    # If we get here, the file was not found.
    raise FileNotFoundError(f"Could not find data file. Checked paths: {potential_paths}")

# --- End of new function ---

def main():
    # Set up argument parser to allow overriding paths from the command line
    parser = argparse.ArgumentParser(description='Run cross-comparative factorial analysis on repair tool data.')
    parser.add_argument('--input', type=str, default='Report/Histories/combined_run_histories.csv', help='Path to the input CSV file.')
    parser.add_argument('--output_dir', type=str, default='Report/Figures', help='Directory to save output figures.')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading data...")
    try:
        data_path = get_data_path(args)
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the 'combined_run_histories.csv' file is in the project root or provide the correct path with --input.")
        return

    # 2. Data Preprocessing
    # Ensure system_prompt names are clean for plotting
    if 'cfg_system_prompt' in df.columns:
        prompt_col = 'cfg_system_prompt'
    elif 'system_prompt' in df.columns:
        prompt_col = 'system_prompt'
    else:
        # Fallback or error if neither is found, though your data seems to have it
        print("Warning: Could not find a system_prompt column. Please check your CSV.")
        return

    # Clean up the system prompt column
    df['system_prompt_clean'] = df[prompt_col].apply(lambda x: str(x).split('/')[-1])
    
    # Convert derived_success to int for calculation
    df['derived_success'] = pd.to_numeric(df['derived_success'], errors='coerce').fillna(0).astype(int)
    
    # 3. Plotting Setup
    sns.set_theme(style="whitegrid")
    
    # Define a custom color palette based on the user's image example
    # Using a light purple for 0 shots and a darker purple/blue for 3 shots
    custom_palette = ["#d8b3e0", "#5c4f8a"]

    # --- Plot 1: Success Rate Factorial Analysis ---
    print("Generating Success Rate Plot...")
    plt.figure(figsize=(12, 6))
    g = sns.catplot(
        data=df, kind="bar",
        x="model", y="derived_success", hue="repair_shots", col="system_prompt_clean",
        height=5, aspect=1.2, errorbar=None, palette=custom_palette
    )
    g.set_axis_labels("Model", "Success Rate")
    g.set_titles("System Prompt: {col_name}")
    g.fig.suptitle("Repair Tool Success Rate by Configuration", y=1.05, fontsize=16)
    g._legend.set_title("Repair Shots")
    
    success_plot_path = os.path.join(args.output_dir, 'success_rate_factorial.png')
    plt.savefig(success_plot_path, bbox_inches='tight', dpi=300)
    plt.close('all')
    print(f"Saved: {success_plot_path}")

    # --- Plot 2: Iterations to Success ---
    print("Generating Iterations to Success Plot...")
    # Filter to only successful runs
    success_df = df[df['derived_success'] == 1].copy()
    
    if success_df.empty:
        print("Warning: No successful runs found. Skipping Iterations to Success plot.")
    else:
        plt.figure(figsize=(12, 6))
        g = sns.catplot(
            data=success_df, kind="box",
            x="model", y="derived_first_success_iteration", hue="repair_shots", col="system_prompt_clean",
            height=5, aspect=1.2, palette=custom_palette
        )
        g.set_axis_labels("Model", "Iterations to Success")
        g.set_titles("System Prompt: {col_name}")
        g.fig.suptitle("Iterations to Success (Successful Runs Only)", y=1.05, fontsize=16)
        g._legend.set_title("Repair Shots")

        iters_plot_path = os.path.join(args.output_dir, 'iterations_to_success.png')
        plt.savefig(iters_plot_path, bbox_inches='tight', dpi=300)
        plt.close('all')
        print(f"Saved: {iters_plot_path}")

    # --- Table: Main Effects Summary ---
    print("Generating Main Effects Summary Table...")
    summary_table = df.groupby(['model', 'system_prompt_clean', 'repair_shots']).agg(
        Total_Runs=('derived_success', 'count'),
        Success_Rate=('derived_success', 'mean'),
        Avg_Iters_to_Success=('derived_first_success_iteration', 'mean')
    ).reset_index()

    # Format the table for readability
    summary_table['Success_Rate_Pct'] = (summary_table['Success_Rate'] * 100).round(2).astype(str) + '%'
    summary_table['Avg_Iters_to_Success'] = summary_table['Avg_Iters_to_Success'].round(2)
    
    # Reorder columns for final output
    summary_table = summary_table[['model', 'system_prompt_clean', 'repair_shots', 'Total_Runs', 'Success_Rate', 'Success_Rate_Pct', 'Avg_Iters_to_Success']]

    table_path = os.path.join(args.output_dir, 'main_effects_summary.csv')
    summary_table.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")
    
    print("\nAnalysis Complete! Summary of the results:")
    print("-" * 50)
    # Print the table without the raw mean for cleaner output
    print(summary_table.drop(columns=['Success_Rate']).to_string(index=False))

if __name__ == "__main__":
    main()
    