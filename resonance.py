import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import pandas as pd

''' Input for 2-delta system can be found around line 474'''

# ============================================================================
# TRANSFER MATRIX FORMULATION
'''PHYSICS SETUP :
- Natural units: ħ = 1, m = 1/2, so E = k²
- Transfer matrix for a single delta at x with strength α:
  M = [[1 + iα/k, i(α/k)exp(-2ikx)],
       [-i(α/k)exp(2ikx), 1 - iα/k]]'''
# ============================================================================

def transfer_matrix_single_delta(k, alpha, x):
    """Transfer matrix for single delta at position x with strength alpha."""
    M11 = 1 + 1j * alpha / k
    M12 = 1j * alpha * np.exp(-2j * k * x) / k
    M21 = -1j * alpha * np.exp(2j * k * x) / k
    M22 = 1 - 1j * alpha / k
    return np.array([[M11, M12], [M21, M22]])

def transfer_matrix_product(k, alphas, positions):
    """Total transfer matrix: M_total = M_N @ ... @ M_2 @ M_1"""
    M_total = np.eye(2, dtype=complex)
    for x_pos, alpha_val in zip(positions, alphas):
        M = transfer_matrix_single_delta(k, alpha_val, x_pos)
        M_total = M @ M_total
    return M_total

def transmission_coefficient(M):
    """Transmission: T = 1/|M11|^2"""
    return 1.0 / np.abs(M[0, 0])**2

def compute_transmission_spectrum(k_values, alphas, positions):
    """Compute T(k) for given configuration"""
    T_values = np.zeros_like(k_values)
    for i, k in enumerate(k_values):
        M = transfer_matrix_product(k, alphas, positions)
        T_values[i] = transmission_coefficient(M)
    return T_values

# ============================================================================
# RESONANCE FINDER FOR 2-DELTA SYSTEM
# ============================================================================

def find_resonance_peaks_analytical(alpha1, alpha2, xa, xb, k_max=3.0, k_min=0.01):
    """
    Find resonance peaks for 2-delta system using analytical formula.
    For α2 = -α1 (equal and opposite), perfect transmission at k_n = πn/Δx
    """
    delta_x = abs(xb - xa)
    
    # Verify assumption
    if not np.isclose(alpha2, -alpha1, rtol=1e-3):
        print(f"⚠️ WARNING: Analytical resonances assume α₂=-α₁. "
              f"Got α₁={alpha1}, α₂={alpha2}. Results may be approximate.")
    
    resonances = []
    n = 1
    while True:
        k_n = np.pi * n / delta_x
        if k_n > k_max:
            break
        if k_n >= k_min:  # Only include resonances above k_min
            resonances.append(k_n)
        n += 1
    
    print(f"\n{'='*70}")
    print(f"ANALYTICAL RESONANCE POSITIONS")
    print(f"{'='*70}")
    print(f"Δx = |x_b - x_a| = {delta_x:.6f}")
    print(f"Formula: k_n = πn/Δx for n = 1, 2, 3, ...")
    print(f"\nResonance peaks ({k_min} < k < {k_max}):")
    for i, k_n in enumerate(resonances, 1):
        print(f"  k_{i} = {k_n:.6f}")
    
    return np.array(resonances)

def compute_window_widths_symmetric(resonance_peaks, k_max=3.0):
    """
    Compute symmetric window widths around each resonance peak.
    All windows have the same width based on spacing between peaks.
    """
    n_peaks = len(resonance_peaks)
    window_ranges = []
    
    # Calculate typical spacing between peaks
    if n_peaks > 1:
        spacings = np.diff(resonance_peaks)
        avg_spacing = np.mean(spacings)
        # Use half the average spacing as the window half-width
        half_width = avg_spacing / 2
    else:
        # Single peak: use distance from k_min to peak
        half_width = resonance_peaks[0] * 0.5
    
    print(f"\n{'='*70}")
    print(f"COMPUTING SYMMETRIC WINDOW RANGES")
    print(f"{'='*70}")
    print(f"Window half-width: {half_width:.6f}")
    
    for i, k_center in enumerate(resonance_peaks):
        k_start = max(0.01, k_center - half_width)  # Don't start below k=0.01
        k_end = min(k_max, k_center + half_width)   # Don't exceed k_max
        
        # Verify resonance is inside window
        if not (k_start < k_center < k_end):
            print(f"  ⚠ WARNING: Resonance {i+1} at k={k_center:.6f} not centered!")
        
        window_ranges.append((k_start, k_end))
        window_width = (k_end - k_start) / 2
        
        print(f"Peak {i+1} at k={k_center:.6f}: range=[{k_start:.6f}, {k_end:.6f}], width={window_width:.6f}")
    
    return window_ranges

# ============================================================================
# CSV SAVING FUNCTION
# ============================================================================

def save_window_result(alpha1_2d, alpha2_2d, xa_2d, xb_2d,
                       k_center, window_width, params_3d, mse_uniform,
                       config_id, window_id, filename='results.csv'):
    """Save one window optimization result to CSV - APPENDS, never overwrites"""
    
    betas_3d, positions_3d = params_to_positions(params_3d, xa_2d, n_deltas_new=3)
    
    delta_x_2d = abs(xb_2d - xa_2d)
    total_strength_2d = abs(alpha1_2d) + abs(alpha2_2d)
    total_strength_3d = sum(betas_3d)
    
    data = {
        'config_id': config_id,
        'window_id': window_id,
        
        # 2-delta input
        'alpha1_2d': alpha1_2d,
        'alpha2_2d': alpha2_2d,
        'xa_2d': xa_2d,
        'xb_2d': xb_2d,
        'delta_x_2d': delta_x_2d,
        'total_strength_2d': total_strength_2d,
        
        # Resonance info
        'k_center': k_center,
        'window_width': window_width,
        'wavelength': 2*np.pi/k_center,
        
        # 3-delta output
        'beta1_3d': betas_3d[0],
        'beta2_3d': betas_3d[1],
        'beta3_3d': betas_3d[2],
        'x1_3d': positions_3d[0],
        'x2_3d': positions_3d[1],
        'x3_3d': positions_3d[2],
        'total_strength_3d': total_strength_3d,
        
        # Derived quantities
        'spacing_12': positions_3d[1] - positions_3d[0],
        'spacing_23': positions_3d[2] - positions_3d[1],
        'total_extent_3d': positions_3d[2] - positions_3d[0],
        
        # Quality metrics
        'mse_uniform': mse_uniform
    }
    
    df = pd.DataFrame([data])
    
    try:
        import os
        file_exists = os.path.isfile(filename)
        
        # APPEND mode - never overwrites
        df.to_csv(filename, mode='a', header=not file_exists, index=False)
        print(f"  ✓ Saved to {filename} (config {config_id}, window {window_id})")
    except Exception as e:
        print(f"  ✗ CSV save failed: {e}")

# ============================================================================
# WINDOWED OPTIMIZATION
# ============================================================================

def params_to_positions(params, x1_fixed, n_deltas_new=3):
    """Convert optimization parameters to ordered positions with fixed x1."""
    betas = params[:n_deltas_new]
    # x1 is now FIXED, not optimized
    delta12 = params[n_deltas_new]
    delta23 = params[n_deltas_new + 1]
    
    x1 = x1_fixed
    x2 = x1 + delta12
    x3 = x2 + delta23
    
    positions = np.array([x1, x2, x3])
    return betas, positions

def get_bounds_for_window(alpha1_2d, n_deltas_new=3, min_separation=0.3):
    """Create bounds for optimization - FIXED upper bound = 2*|alpha1_2d|.
    Note: x1 is now FIXED, so no bounds needed for it."""
    strength_upper =  2*abs(alpha1_2d)  # FIXED upper bound
    
    bounds = []
    
    # Bounds for betas - POSITIVE ONLY with FIXED upper bound
    for i in range(n_deltas_new):
        bounds.append((0.5, strength_upper))  # Positive strengths only, minimum 0.5
    
    # Bounds for spacing (x1 is fixed, only optimize delta12 and delta23)
    bounds.append((min_separation, 5.0))  # delta12
    bounds.append((min_separation, 5.0))  # delta23
    
    return bounds

def optimize_window(alpha1_2d, alpha2_2d, xa_2d, xb_2d, 
                   k_range,
                   min_strength=0.5, min_separation=0.3):
    """Optimize 3-delta system for a single resonance window with FIXED x1."""
    
    k_min, k_max = k_range
    k_center = (k_min + k_max) / 2
    window_width = (k_max - k_min) / 2
    
    k_vals = np.linspace(k_min, k_max, 200)
    alphas_2d = np.array([alpha1_2d, alpha2_2d])
    positions_2d = np.array([xa_2d, xb_2d])
    
    # Compute target spectrum
    T_target = compute_transmission_spectrum(k_vals, alphas_2d, positions_2d)
    
    # Get bounds with FIXED upper bound
    bounds = get_bounds_for_window(alpha1_2d, n_deltas_new=3, 
                                   min_separation=min_separation)
    
    strength_upper = 2.0 * abs(alpha1_2d)
    
    print(f"\n{'='*70}")
    print(f"WINDOW OPTIMIZATION: k ∈ [{k_min:.3f}, {k_max:.3f}] (centered at k={k_center:.3f}, width={window_width:.3f})")
    print(f"x1 FIXED at {xa_2d:.6f} (translational invariance)")
    print(f"Strength bounds: β ∈ [0.5, {strength_upper:.2f}] (FIXED upper bound = 2×|α₁|)")
    print(f"{'='*70}")
    
    def objective(params):
        betas_3d, positions_3d = params_to_positions(params, xa_2d, n_deltas_new=3)
        
        # Enforce minimum strength (already in bounds, but double-check)
        if any(beta < min_strength for beta in betas_3d):
            return 1e10
        
        try:
            T_3d = compute_transmission_spectrum(k_vals, betas_3d, positions_3d)
            
            # Uniform MSE (no weighting)
            mse = np.mean((T_target - T_3d)**2)
            return mse
        except:
            return 1e10
    
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=800,
        popsize=20,
        atol=1e-10,
        tol=1e-10,
        workers=1,
        disp=False
    )
    
    betas_3d, positions_3d = params_to_positions(result.x, xa_2d, n_deltas_new=3)
    
    # Calculate uniform MSE only
    T_3d = compute_transmission_spectrum(k_vals, betas_3d, positions_3d)
    mse_uniform = np.mean((T_target - T_3d)**2)
    
    print(f"✓ Uniform MSE = {mse_uniform:.6e}")
    print(f"3-δ solution (POSITIVE STRENGTHS, x1={xa_2d:.6f} fixed):")
    for i, (x, b) in enumerate(zip(positions_3d, betas_3d)):
        print(f"  δ{i+1}: β = {b:+.6f} at x = {x:+.6f}")
    
    return result.x, k_vals, mse_uniform

# ============================================================================
# PLOTTING
# ============================================================================

def plot_windowed_results(alpha1_2d, alpha2_2d, xa_2d, xb_2d, 
                         resonance_peaks, window_results, window_ranges):
    """
    Create comprehensive plot showing windowed optimization with analysis panels.
    Top: Full spectrum
    Bottom Left: Individual strength bar chart
    Bottom Right: Spatial configuration diagram (schematic with true proportions)
    """
    alphas_2d = np.array([alpha1_2d, alpha2_2d])
    positions_2d = np.array([xa_2d, xb_2d])
    
    # Full k-range for reference
    k_full = np.linspace(0.01, 3.0, 500)
    T_2d_full = compute_transmission_spectrum(k_full, alphas_2d, positions_2d)
    
    # Create figure with 2 rows
    fig = plt.figure(figsize=(18, 10))
    
    # Define DISTINCT colors for each window (better color separation)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#5DADE2', '#F8B739', '#EB984E']
    
    # ========== TOP PANEL: Full spectrum with windows highlighted ==========
    ax_top = plt.subplot(2, 2, (1, 2))  # Spans both columns
    
    # Plot full 2-delta spectrum with combined label
    ax_top.plot(k_full, T_2d_full, 'b-', linewidth=2.5, 
               label=f'2-δ system: α₁={alpha1_2d:.2f}, α₂={alpha2_2d:.2f}', zorder=3)
    
    # Highlight each window and plot optimized 3-delta
    for i, (k_center, params_3d, k_vals, mse_u) in enumerate(window_results):
        color = colors[i % len(colors)]
        
        # Shade window region
        ax_top.axvspan(k_vals[0], k_vals[-1], alpha=0.15, color=color, zorder=1)
        
        # Plot 3-delta spectrum in this window
        betas_3d, positions_3d = params_to_positions(params_3d, xa_2d, n_deltas_new=3)
        T_3d_window = compute_transmission_spectrum(k_vals, betas_3d, positions_3d)
        ax_top.plot(k_vals, T_3d_window, '--', color=color, linewidth=2.5, 
                   label=f'3-δ Win {i+1} (MSE_u={mse_u:.2e})', zorder=4)
    
    ax_top.set_xlabel('Wave number k', fontsize=13, fontweight='bold')
    ax_top.set_ylabel('Transmission T(k)', fontsize=13, fontweight='bold')
    ax_top.set_title('Windowed Optimization: Full Spectrum View', 
                     fontsize=15, fontweight='bold')
    ax_top.legend(loc='upper left', fontsize=9, ncol=2)
    ax_top.grid(True, alpha=0.3, linestyle='--')
    ax_top.set_xlim([0, 3.0])
    ax_top.set_ylim([0, 1.05])
    
    # ========== BOTTOM LEFT: Individual Strength Bar Chart ==========
    ax_strength = plt.subplot(2, 2, 3)
    
    n_windows = len(window_results)
    bar_width = 0.25
    x_positions = np.arange(n_windows)
    
    # Extract strength values for each window
    beta1_vals = []
    beta2_vals = []
    beta3_vals = []
    
    for k_center, params_3d, k_vals, mse_u in window_results:
        betas_3d, _ = params_to_positions(params_3d, xa_2d, n_deltas_new=3)
        beta1_vals.append(betas_3d[0])
        beta2_vals.append(betas_3d[1])
        beta3_vals.append(betas_3d[2])
    
    # Create grouped bar chart for 3-delta (NO 2-delta reference lines here)
    ax_strength.bar(x_positions - bar_width, beta1_vals, bar_width, 
                   label='3-δ: β₁', color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax_strength.bar(x_positions, beta2_vals, bar_width, 
                   label='3-δ: β₂', color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax_strength.bar(x_positions + bar_width, beta3_vals, bar_width, 
                   label='3-δ: β₃', color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax_strength.set_xlabel('Window Number', fontsize=12, fontweight='bold')
    ax_strength.set_ylabel('Strength', fontsize=12, fontweight='bold')
    ax_strength.set_title('Individual Strength Distribution (3-δ only)', fontsize=13, fontweight='bold')
    ax_strength.set_xticks(x_positions)
    ax_strength.set_xticklabels([f'W{i+1}' for i in range(n_windows)])
    ax_strength.legend(fontsize=9, loc='upper left', ncol=1)
    ax_strength.grid(True, alpha=0.3, axis='y')
    y_max = max(max(beta1_vals), max(beta2_vals), max(beta3_vals))
    ax_strength.set_ylim([0, y_max * 1.2])
    
    # Add resonance k values as text ABOVE the x-axis label
    for i, k_center in enumerate(resonance_peaks):
        ax_strength.text(i, -0.15 * y_max, f'k≈{k_center:.2f}', 
                        ha='center', fontsize=8, style='italic', color='gray')
    
    # ========== BOTTOM RIGHT: Spatial Configuration Schematic ==========
    ax_spatial = plt.subplot(2, 2, 4)
    
    # Find global x-range for proper scaling
    all_positions = [xa_2d, xb_2d]
    for k_center, params_3d, k_vals, mse_u in window_results:
        _, positions_3d = params_to_positions(params_3d, xa_2d, n_deltas_new=3)
        all_positions.extend(positions_3d)
    
    x_min = min(all_positions)
    x_max = max(all_positions)
    x_range = x_max - x_min
    x_margin = x_range * 0.1
    
    # Plot 2-delta reference
    y_2d = 0
    ax_spatial.plot([xa_2d, xb_2d], [y_2d, y_2d], 'b-', linewidth=3, alpha=0.5, zorder=2)
    ax_spatial.scatter([xa_2d, xb_2d], [y_2d, y_2d], color='blue', s=250, marker='D', 
                      edgecolors='darkblue', linewidths=2.5, zorder=5, alpha=0.8)
    
    # Add 2-delta labels (MOVED UP to avoid overlap with diamonds)
    ax_spatial.text(xa_2d, y_2d + 0.25, f'x_a={xa_2d:.2f}\nα₁={alpha1_2d:.2f}', 
                   ha='center', fontsize=8, color='darkblue',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.6))
    ax_spatial.text(xb_2d, y_2d + 0.25, f'x_b={xb_2d:.2f}\nα₂={alpha2_2d:.2f}', 
                   ha='center', fontsize=8, color='darkblue',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.6))
    
    # Label 2-delta system
    ax_spatial.text(x_min - x_margin * 0.5, y_2d, '2-δ', ha='right', va='center', 
                   fontsize=11, fontweight='bold', color='blue')
    
    # Plot each 3-delta configuration
    vertical_spacing = 0.8
    for i, (k_center, params_3d, k_vals, mse_u) in enumerate(window_results):
        betas_3d, positions_3d = params_to_positions(params_3d, xa_2d, n_deltas_new=3)
        color = colors[i % len(colors)]
        
        # Vertical position
        y_offset = (i + 1) * vertical_spacing
        
        # Plot connecting line
        ax_spatial.plot(positions_3d, [y_offset]*3, color=color, linewidth=3, 
                       alpha=0.5, zorder=2)
        
        # Plot positions as scatter points
        ax_spatial.scatter(positions_3d, [y_offset]*3, color=color, s=200, 
                          marker='o', edgecolors='black', linewidths=2, zorder=4)
        
        # Add position and strength labels
        for j, (x, b) in enumerate(zip(positions_3d, betas_3d)):
            ax_spatial.text(x, y_offset + 0.15, f'x{j+1}={x:.2f}\nβ={b:.2f}', 
                          ha='center', fontsize=7, color='black',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.4))
        
        # Label window
        ax_spatial.text(x_min - x_margin * 0.5, y_offset, f'W{i+1}', ha='right', va='center', 
                       fontsize=11, fontweight='bold', color=color)
    
    ax_spatial.set_xlabel('Position x (true scale)', fontsize=12, fontweight='bold')
    ax_spatial.set_ylabel('Configuration', fontsize=12, fontweight='bold')
    ax_spatial.set_title('Spatial Configuration Schematic (x₁ fixed, true proportions)', fontsize=13, fontweight='bold')
    ax_spatial.set_xlim([x_min - x_margin, x_max + x_margin])
    ax_spatial.set_ylim([-0.5, (n_windows + 0.5) * vertical_spacing])
    ax_spatial.grid(True, alpha=0.3, axis='x')
    ax_spatial.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('windowed_optimization_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("✅ Figure saved: windowed_optimization_analysis.pdf")
    print("="*70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("WINDOWED RESONANCE MATCHING: 2-DELTA → 3-DELTA (FIXED UPPER BOUND, FIXED x1)")
    print("="*70)
    
    # ========================================================================
    # INPUT: Your 2-delta system (CHANGE THESE TO TEST DIFFERENT CONFIGS)
    # ========================================================================
    
    # CONFIG ID - increment this manually each time you run with new parameters
    CONFIG_ID = 1.2  # <-- CHANGE THIS for each new run. For better conveneinace with result_analysis.py, change it accordingly. 
                            # Example: For setup with |alpha| = 1 and 2 resonances within k<3, CONFIG ID = 1.2
              
    
    alpha1_2d = 4
    alpha2_2d = -4
    xa_2d = 0.0
    xb_2d = 5.95
    
    strength_upper = 2* abs(alpha1_2d)
    
    print(f"\n2-δ Input System (Config ID = {CONFIG_ID}):")
    print(f"  α₁ = {alpha1_2d:.6f}, α₂ = {alpha2_2d:.6f}")
    print(f"  x_a = {xa_2d:.6f}, x_b = {xb_2d:.6f}")
    print(f"  Note: x₁ will be FIXED for all 3-δ optimizations (translational invariance)")
    print(f"  Strength constraint: β ∈ [0.5, {strength_upper:.2f}] (FIXED upper bound = 2×|α₁|)")
    
    # ========================================================================
    # STEP 1: Find resonance peaks analytically
    # ========================================================================
    resonance_peaks = find_resonance_peaks_analytical(alpha1_2d, alpha2_2d, xa_2d, xb_2d, k_max=3.0, k_min=0.01)
    
    # ========================================================================
    # STEP 2: Compute symmetric window ranges
    # ========================================================================
    window_ranges = compute_window_widths_symmetric(resonance_peaks, k_max=3.0)
    
    # ========================================================================
    # STEP 3: Optimize 3-delta system for each window
    # ========================================================================
    window_results = []
    
    print(f"\n{'='*70}")
    print(f"PERFORMING WINDOWED OPTIMIZATIONS (FIXED UPPER BOUND = 2×|α₁| = {strength_upper:.2f}, FIXED x1={xa_2d:.6f})")
    print(f"{'='*70}")
    
    for i, (k_center, k_range) in enumerate(zip(resonance_peaks, window_ranges)):
        k_min, k_max = k_range
        window_width = (k_max - k_min) / 2
        
        print(f"\n>>> Processing resonance peak {i+1}/{len(resonance_peaks)} at k = {k_center:.6f}")
        print(f"    Window range: [{k_min:.6f}, {k_max:.6f}], width = {window_width:.6f}")
        
        params_3d, k_vals, mse_uniform = optimize_window(
            alpha1_2d, alpha2_2d, xa_2d, xb_2d,
            k_range=k_range,
            min_strength=0.5,
            min_separation=0.3
        )
        
        # SAVE TO CSV - APPENDS to existing file
        save_window_result(
            alpha1_2d, alpha2_2d, xa_2d, xb_2d,
            k_center, window_width, params_3d, mse_uniform,
            config_id=CONFIG_ID, window_id=i
        )
        
        window_results.append((k_center, params_3d, k_vals, mse_uniform))
    
    # ========================================================================
    # STEP 4: Plot results
    # ========================================================================
    plot_windowed_results(alpha1_2d, alpha2_2d, xa_2d, xb_2d, 
                         resonance_peaks, window_results, window_ranges)
    
    # ========================================================================
    # STEP 5: Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY OF WINDOWED OPTIMIZATIONS")
    print(f"{'='*70}")
    
    for i, (k_center, params_3d, k_vals, mse_u) in enumerate(window_results):
        betas_3d, positions_3d = params_to_positions(params_3d, xa_2d, n_deltas_new=3)
        k_min, k_max = window_ranges[i]
        window_width = (k_max - k_min) / 2
        
        print(f"\nWindow {i+1} (centered at k = {k_center:.6f}, range = [{k_min:.6f}, {k_max:.6f}]):")
        print(f"  Width: {window_width:.6f}")
        print(f"  Uniform MSE: {mse_u:.6e}")
        print(f"  3-δ system (POSITIVE STRENGTHS ∈ [0.5, {strength_upper:.2f}], x1={xa_2d:.6f} fixed):")
        for j, (x, b) in enumerate(zip(positions_3d, betas_3d)):
            print(f"    δ{j+1}: β = {b:+.6f} at x = {x:+.6f}")
    
    print("\n" + "="*70)
    print("✅ All windowed optimizations complete!")
    print(f"✅ Results saved to results.csv (Config ID = {CONFIG_ID})")
    print(f"✅ Figure saved as PDF: windowed_optimization_analysis.pdf")
    print("="*70)
