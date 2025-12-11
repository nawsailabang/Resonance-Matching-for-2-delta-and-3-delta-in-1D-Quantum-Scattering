import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('results.csv')

# Fix the duplicate 3.4 entries - keep only first 4
df_34 = df[df['config_id'] == 3.4]
if len(df_34) > 4:
    # Get indices of the last 4 rows with config_id == 3.4
    indices_to_drop = df_34.tail(4).index
    df = df.drop(indices_to_drop)

# Extract alpha value from config_id (e.g., 2.3 -> alpha=2, 3.4 -> alpha=3)
df['alpha'] = df['config_id'].apply(lambda x: int(str(x).split('.')[0]))

# ===== Scaling Analysis by Alpha =====
scaling_data = []
for alpha in df['alpha'].unique():
    df_alpha = df[df['alpha'] == alpha]
    for config_id in df_alpha['config_id'].unique():
        config_df = df_alpha[df_alpha['config_id'] == config_id]
        num_windows = len(config_df)
        avg_mse = config_df['mse_uniform'].mean()
        
        scaling_data.append({
            'alpha': alpha,
            'config_id': config_id,
            'num_windows': num_windows,
            'avg_mse': avg_mse
        })

scaling_df = pd.DataFrame(scaling_data)

# ===== Window-by-Window Analysis by Alpha =====
window_data = []
for alpha in df['alpha'].unique():
    df_alpha = df[df['alpha'] == alpha]
    window_analysis = df_alpha.groupby('window_id').agg({
        'mse_uniform': 'mean'
    })
    
    for window_id in window_analysis.index:
        window_data.append({
            'alpha': alpha,
            'window_id': window_id,
            'mse_mean': window_analysis.loc[window_id, 'mse_uniform']
        })

window_df = pd.DataFrame(window_data)

# ===== Create Overlaid Plots =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Define colors and labels for each alpha
colors = {2: 'steelblue', 3: 'coral', 4: 'seagreen'}
labels = {2: 'Approximating α₁,₂ = ±2', 
          3: 'Approximating α₁,₂ = ±3', 
          4: 'Approximating α₁,₂ = ±4'}

# Plot 1: MSE vs Number of Resonances (overlaid scatter with same marker)
for alpha in sorted(df['alpha'].unique()):
    alpha_data = scaling_df[scaling_df['alpha'] == alpha]
    axes[0].scatter(alpha_data['num_windows'], alpha_data['avg_mse'], 
                   s=120, color=colors[alpha], marker='o', 
                   label=labels[alpha], alpha=0.7, edgecolors='black', linewidth=0.5)

axes[0].set_xlabel('Number of Resonances', fontsize=12)
axes[0].set_ylabel('Average MSE', fontsize=12)
axes[0].set_yscale('log')
axes[0].set_title('MSE Scaling with Resonance Count\n(3-δ approximation quality)', 
                  fontsize=13, fontweight='bold')
axes[0].set_xticks(range(1, 6))
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10, loc='best')

# Plot 2: MSE by Resonance Position (grouped bars)
resonance_labels = ['1st\nResonances', '2nd\nResonances', '3rd\nResonances', 
                   '4th\nResonances', '5th\nResonances']
x_positions = np.arange(5)
bar_width = 0.25

for i, alpha in enumerate(sorted(df['alpha'].unique())):
    alpha_data = window_df[window_df['alpha'] == alpha]
    
    # Create array with MSE values, filling missing positions with NaN
    mse_values = []
    for window_id in range(5):
        window_data = alpha_data[alpha_data['window_id'] == window_id]
        if len(window_data) > 0:
            mse_values.append(window_data['mse_mean'].values[0])
        else:
            mse_values.append(np.nan)
    
    # Plot bars with offset
    offset = bar_width * (i - 1)
    axes[1].bar(x_positions + offset, mse_values, bar_width,
               color=colors[alpha], alpha=0.7, label=labels[alpha],
               edgecolor='black', linewidth=0.5)

axes[1].set_xlabel('Resonance Position', fontsize=12)
axes[1].set_ylabel('Average MSE', fontsize=12)
axes[1].set_yscale('log')
axes[1].set_title('MSE by Resonance Position\n(3-δ approximation quality)', 
                  fontsize=13, fontweight='bold')
axes[1].set_xticks(x_positions)
axes[1].set_xticklabels(resonance_labels, fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].legend(fontsize=10, loc='best')

plt.tight_layout()
plt.savefig('mse_analysis.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete! Plot saved to 'mse_analysis.pdf'")
print("\nSummary by coupling strength:")
for alpha in sorted(df['alpha'].unique()):
    alpha_data = window_df[window_df['alpha'] == alpha]
    if len(alpha_data) > 0:
        first_res = alpha_data[alpha_data['window_id'] == 0]['mse_mean'].values
        if len(first_res) > 0:
            print(f"Approximating α₁,₂ = ±{alpha}: 1st resonance MSE = {first_res[0]:.2e}")