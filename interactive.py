"""
Interactive Isospectral Explorer for Dirac Delta Potentials

This tool allows interactive exploration of near-isospectral configurations
between two-delta and three-delta quantum systems in 1D scattering.
In the direction of paper, this tool could be used for desinging the 2-delta system.

PHYSICS SETUP :
- Natural units: ħ = 1, m = 1/2, so E = k²
- Transfer matrix for a single delta at x with strength α:
  M = [[1 + iα/k, i(α/k)exp(-2ikx)],
       [-i(α/k)exp(2ikx), 1 - iα/k]]
- Total matrix: M_total = M_N ⋯ M₂ M₁ (right-multiply for increasing x)
- Transmission: T(k) = 1/|(M_total)₁₁|²
- FOCUSED ON NON-PERTURBATIVE REGIME: 0 < k < 3 (E < 9 in natural units)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec

def create_isospectral_explorer_wide_range():
    """Interactive explorer for finding near-isospectral configurations"""
    
    # Initial parameters - more diverse starting point
    init_params = {
        'alpha1': 2, 'alpha2': -2,  # Two-delta strengths
        'x1_dd': 0, 'x2_dd': 1.0,    # Two-delta positions (x1_dd < x2_dd)
        'beta1': 0.5, 'beta2': -2.0, 'beta3': 1.8,  # Three-delta strengths
        'x1_td': -1.8, 'x2_td': 0.5, 'x3_td': 2.2  # Three-delta positions (x1_td < x2_td < x3_td)
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 2, height_ratios=[0.6, 2, 2, 1], hspace=0.3)
    
    # Parameter box at the top
    ax_param_display = fig.add_subplot(gs[0, :])
    
    # Main plots
    ax_trans = fig.add_subplot(gs[1, :])
    ax_potential = fig.add_subplot(gs[2, :])
    ax_diff = fig.add_subplot(gs[3, :])
    
    plt.subplots_adjust(bottom=0.5)
    
    # Wave number range extended to k < 3
    k_plot = np.linspace(0.1, 3.0, 600)
    
    def compute_transmission(params):
        """Compute transmission for both systems - matches your paper's equations"""
        T_DDDP, T_TDDP = [], []
        
        for k in k_plot:
            # -------------------------------------------------
            # TWO-DELTA SYSTEM (Equation matches paper's natural units)
            # -------------------------------------------------
            beta1 = params['alpha1'] / k  # α₁/k
            beta2 = params['alpha2'] / k  # α₂/k
            
            # Sort by position for correct matrix multiplication
            dd_positions = [(params['x1_dd'], beta1), (params['x2_dd'], beta2)]
            dd_positions.sort(key=lambda x: x[0])
            
            M_total_dd = np.eye(2, dtype=complex)
            for x_pos, beta_val in dd_positions:
                # Single delta matrix from paper Eq. (4) with η = α/k
                M = np.array([
                    [1 + 1j*beta_val, 1j*beta_val * np.exp(-2j*k*x_pos)],
                    [-1j*beta_val * np.exp(2j*k*x_pos), 1 - 1j*beta_val]
                ])
                M_total_dd = M @ M_total_dd  # Right-multiply for increasing x
            
            T_DDDP.append(1 / np.abs(M_total_dd[0,0])**2)
            
            # -------------------------------------------------
            # THREE-DELTA SYSTEM
            # -------------------------------------------------
            beta1_t = params['beta1'] / k  # β₁/k
            beta2_t = params['beta2'] / k  # β₂/k
            beta3_t = params['beta3'] / k  # β₃/k
            
            td_positions = [
                (params['x1_td'], beta1_t), 
                (params['x2_td'], beta2_t), 
                (params['x3_td'], beta3_t)
            ]
            td_positions.sort(key=lambda x: x[0])
            
            M_total_td = np.eye(2, dtype=complex)
            for x_pos, beta_val in td_positions:
                M = np.array([
                    [1 + 1j*beta_val, 1j*beta_val * np.exp(-2j*k*x_pos)],
                    [-1j*beta_val * np.exp(2j*k*x_pos), 1 - 1j*beta_val]
                ])
                M_total_td = M @ M_total_td
            
            T_TDDP.append(1 / np.abs(M_total_td[0,0])**2)
        
        return np.array(T_DDDP), np.array(T_TDDP)
    
    # Initial plot
    T_DDDP, T_TDDP = compute_transmission(init_params)
    diff = T_DDDP - T_TDDP
    
    # Transmission plot
    line1, = ax_trans.plot(k_plot, T_DDDP, 'b-', linewidth=2, label='Two-delta')
    line2, = ax_trans.plot(k_plot, T_TDDP, 'r--', linewidth=2, label='Three-delta')
    ax_trans.set_xlabel('Wave number k (ħ=1, m=1/2 → E=k²)', fontsize=12)
    ax_trans.set_ylabel('Transmission T(k)', fontsize=12)
    ax_trans.legend(fontsize=11)
    ax_trans.grid(True, alpha=0.3)
    ax_trans.set_ylim(-0.1, 1.1)
    ax_trans.set_xlim(0, 3)
    
    # Potential plot
    ax_potential.clear()
    ax_potential.stem([init_params['x1_dd'], init_params['x2_dd']], 
                     [init_params['alpha1'], init_params['alpha2']], 
                     linefmt='b-', markerfmt='bo', basefmt=' ', label='Two-delta')
    ax_potential.stem([init_params['x1_td'], init_params['x2_td'], init_params['x3_td']],
                     [init_params['beta1'], init_params['beta2'], init_params['beta3']],
                     linefmt='r-', markerfmt='rs', basefmt=' ', label='Three-delta')
    ax_potential.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax_potential.set_xlabel('Position x')
    ax_potential.set_ylabel('Potential Strength')
    ax_potential.legend()
    ax_potential.grid(True, alpha=0.3)
    ax_potential.set_xlim(-6, 6)
    ax_potential.set_ylim(-10, 10)
    
    # Add value labels
    label_offset = 1.0
    for x, y, color in zip([init_params['x1_dd'], init_params['x2_dd']], 
                          [init_params['alpha1'], init_params['alpha2']], 
                          ['blue', 'blue']):
        va = 'bottom' if y > 0 else 'top'
        y_pos = y + label_offset if y > 0 else y - label_offset
        ax_potential.text(x, y_pos, f'{y:.2f}', 
                         ha='center', va=va, 
                         color=color, weight='bold', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    for x, y, color in zip([init_params['x1_td'], init_params['x2_td'], init_params['x3_td']],
                          [init_params['beta1'], init_params['beta2'], init_params['beta3']], 
                          ['red', 'red', 'red']):
        va = 'bottom' if y > 0 else 'top'
        y_pos = y + label_offset if y > 0 else y - label_offset
        ax_potential.text(x, y_pos, f'{y:.2f}', 
                         ha='center', va=va, 
                         color=color, weight='bold', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Difference plot
    line_diff, = ax_diff.plot(k_plot, diff, 'g-', linewidth=1.5)
    ax_diff.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax_diff.set_xlabel('Wave number k', fontsize=12)
    ax_diff.set_ylabel('ΔT(k)', fontsize=12)
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_ylim(-1, 1)
    ax_diff.set_xlim(0, 3)
    
    # Error metrics
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))
    error_text = f'Max difference: {max_diff:.3f}\nRMS difference: {rms_diff:.3f}'
    error_text_obj = ax_diff.text(0.02, 0.98, error_text, transform=ax_diff.transAxes, 
                    va='top', ha='left', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Parameter display box
    ax_param_display.axis('off')
    
    def format_param_tuple(params):
        """Format parameters as a Python tuple for easy copying"""
        return f"({params['alpha1']:.4f}, {params['alpha2']:.4f}, " \
               f"{params['x1_dd']:.4f}, {params['x2_dd']:.4f}, " \
               f"{params['beta1']:.4f}, {params['beta2']:.4f}, {params['beta3']:.4f}, " \
               f"{params['x1_td']:.4f}, {params['x2_td']:.4f}, {params['x3_td']:.4f})"
    
    param_text = format_param_tuple(init_params)
    param_display_text = ax_param_display.text(0.5, 0.5, 
        f"Parameters (α1, α2, x1_dd, x2_dd, β1, β2, β3, x1_td, x2_td, x3_td):\n{param_text}", 
        transform=ax_param_display.transAxes,
        fontsize=12, fontfamily='monospace',
        va='center', ha='center',
        bbox=dict(boxstyle="round,pad=0.8", 
                 facecolor="lightyellow", 
                 edgecolor="black", 
                 linewidth=1,
                 alpha=0.9))
    
    # Create sliders
    slider_y = 0.35
    slider_height = 0.02
    slider_spacing = 0.03
    
    ax_alpha1 = plt.axes([0.25, slider_y, 0.65, slider_height])
    ax_alpha2 = plt.axes([0.25, slider_y - slider_spacing, 0.65, slider_height])
    ax_x1_dd = plt.axes([0.25, slider_y - 2*slider_spacing, 0.65, slider_height])
    ax_x2_dd = plt.axes([0.25, slider_y - 3*slider_spacing, 0.65, slider_height])
    ax_beta1 = plt.axes([0.25, slider_y - 4*slider_spacing, 0.65, slider_height])
    ax_beta2 = plt.axes([0.25, slider_y - 5*slider_spacing, 0.65, slider_height])
    ax_beta3 = plt.axes([0.25, slider_y - 6*slider_spacing, 0.65, slider_height])
    ax_x1_td = plt.axes([0.25, slider_y - 7*slider_spacing, 0.65, slider_height])
    ax_x2_td = plt.axes([0.25, slider_y - 8*slider_spacing, 0.65, slider_height])
    ax_x3_td = plt.axes([0.25, slider_y - 9*slider_spacing, 0.65, slider_height])
    
    sliders = {
        'alpha1': Slider(ax_alpha1, 'α₁ (2-delta)', -10.0, 10.0, valinit=init_params['alpha1']),
        'alpha2': Slider(ax_alpha2, 'α₂ (2-delta)', -10.0, 10.0, valinit=init_params['alpha2']),
        'x1_dd': Slider(ax_x1_dd, 'x₁ (2-delta)', -6.0, 6.0, valinit=init_params['x1_dd']),
        'x2_dd': Slider(ax_x2_dd, 'x₂ (2-delta)', -6.0, 6.0, valinit=init_params['x2_dd']),
        'beta1': Slider(ax_beta1, 'β₁ (3-delta)', -10.0, 10.0, valinit=init_params['beta1']),
        'beta2': Slider(ax_beta2, 'β₂ (3-delta)', -10.0, 10.0, valinit=init_params['beta2']),
        'beta3': Slider(ax_beta3, 'β₃ (3-delta)', -10.0, 10.0, valinit=init_params['beta3']),
        'x1_td': Slider(ax_x1_td, 'x₁ (3-delta)', -6.0, 6.0, valinit=init_params['x1_td']),
        'x2_td': Slider(ax_x2_td, 'x₂ (3-delta)', -6.0, 6.0, valinit=init_params['x2_td']),
        'x3_td': Slider(ax_x3_td, 'x₃ (3-delta)', -6.0, 6.0, valinit=init_params['x3_td'])
    }
    
    # Color position slider labels
    sliders['x1_dd'].label.set_color('blue')
    sliders['x2_dd'].label.set_color('blue')
    sliders['x1_td'].label.set_color('red')
    sliders['x2_td'].label.set_color('red')
    sliders['x3_td'].label.set_color('red')
    
    # Track last valid positions
    last_valid = init_params.copy()
    
    def check_and_correct_constraints(params, slider_name):
        """Enforce position ordering constraints"""
        corrected_params = params.copy()
        
        if slider_name in ['x1_dd', 'x2_dd']:
            x1 = corrected_params['x1_dd']
            x2 = corrected_params['x2_dd']
            
            if slider_name == 'x1_dd':
                if x1 >= x2:
                    corrected_params['x1_dd'] = x2 - 0.01
            elif slider_name == 'x2_dd':
                if x2 <= x1:
                    corrected_params['x2_dd'] = x1 + 0.01
        
        elif slider_name in ['x1_td', 'x2_td', 'x3_td']:
            x1 = corrected_params['x1_td']
            x2 = corrected_params['x2_td']
            x3 = corrected_params['x3_td']
            
            if slider_name == 'x1_td':
                if x1 >= x2:
                    corrected_params['x1_td'] = x2 - 0.01
            elif slider_name == 'x2_td':
                if x2 <= x1:
                    corrected_params['x2_td'] = x1 + 0.01
                elif x2 >= x3:
                    corrected_params['x2_td'] = x3 - 0.01
            elif slider_name == 'x3_td':
                if x3 <= x2:
                    corrected_params['x3_td'] = x2 + 0.01
        
        return corrected_params
    
    def make_update_func(slider_name):
        def update(val):
            current_params = {name: s.val for name, s in sliders.items()}
            corrected_params = check_and_correct_constraints(current_params, slider_name)
            
            if abs(corrected_params[slider_name] - current_params[slider_name]) > 0.001:
                sliders[slider_name].set_val(corrected_params[slider_name])
                return
            
            last_valid[slider_name] = corrected_params[slider_name]
            
            # Recompute
            T_DDDP, T_TDDP = compute_transmission(corrected_params)
            diff = T_DDDP - T_TDDP
            
            # Update plots
            line1.set_ydata(T_DDDP)
            line2.set_ydata(T_TDDP)
            line_diff.set_ydata(diff)
            
            ax_potential.clear()
            ax_potential.stem([corrected_params['x1_dd'], corrected_params['x2_dd']], 
                             [corrected_params['alpha1'], corrected_params['alpha2']], 
                             linefmt='b-', markerfmt='bo', basefmt=' ', label='Two-delta')
            ax_potential.stem([corrected_params['x1_td'], corrected_params['x2_td'], corrected_params['x3_td']],
                             [corrected_params['beta1'], corrected_params['beta2'], corrected_params['beta3']],
                             linefmt='r-', markerfmt='rs', basefmt=' ', label='Three-delta')
            ax_potential.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax_potential.set_xlabel('Position x')
            ax_potential.set_ylabel('Potential Strength')
            ax_potential.legend()
            ax_potential.grid(True, alpha=0.3)
            ax_potential.set_xlim(-6, 6)
            ax_potential.set_ylim(-10, 10)
            
            # Update labels
            for x, y, color in zip([corrected_params['x1_dd'], corrected_params['x2_dd']], 
                                  [corrected_params['alpha1'], corrected_params['alpha2']], 
                                  ['blue', 'blue']):
                va = 'bottom' if y > 0 else 'top'
                y_pos = y + label_offset if y > 0 else y - label_offset
                ax_potential.text(x, y_pos, f'{y:.2f}', 
                                 ha='center', va=va, 
                                 color=color, weight='bold', fontsize=9,
                                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            for x, y, color in zip([corrected_params['x1_td'], corrected_params['x2_td'], corrected_params['x3_td']],
                                  [corrected_params['beta1'], corrected_params['beta2'], corrected_params['beta3']], 
                                  ['red', 'red', 'red']):
                va = 'bottom' if y > 0 else 'top'
                y_pos = y + label_offset if y > 0 else y - label_offset
                ax_potential.text(x, y_pos, f'{y:.2f}', 
                                 ha='center', va=va, 
                                 color=color, weight='bold', fontsize=9,
                                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            # Update error text
            max_diff = np.max(np.abs(diff))
            rms_diff = np.sqrt(np.mean(diff**2))
            error_text = f'Max difference: {max_diff:.3f}\nRMS difference: {rms_diff:.3f}'
            ax_diff.texts[0].remove()
            ax_diff.text(0.02, 0.98, error_text, transform=ax_diff.transAxes, 
                        va='top', ha='left', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Update parameter display
            param_text = format_param_tuple(corrected_params)
            param_display_text.set_text(
                f"Parameters (α1, α2, x1_dd, x2_dd, β1, β2, β3, x1_td, x2_td, x3_td):\n{param_text}"
            )
            
            fig.canvas.draw_idle()
        
        return update
    
    # Connect sliders
    for name, slider in sliders.items():
        slider.on_changed(make_update_func(name))
    
    # Add reset button
    resetax = plt.axes([0.8, 0.02, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    
    def reset(event):
        for name, slider in sliders.items():
            slider.set_val(init_params[name])
    
    button.on_clicked(reset)
    
    # Add preset button
    preset_ax1 = plt.axes([0.1, 0.02, 0.15, 0.04])
    preset_btn1 = Button(preset_ax1, 'Preset: Symmetric')
    
    def load_preset1(event):
        sliders['alpha1'].set_val(1.0)
        sliders['alpha2'].set_val(-1.0)
        sliders['x1_dd'].set_val(-1.0)
        sliders['x2_dd'].set_val(1.0)
        sliders['beta1'].set_val(0.8)
        sliders['beta2'].set_val(-0.5)
        sliders['beta3'].set_val(0.7)
        sliders['x1_td'].set_val(-1.5)
        sliders['x2_td'].set_val(0.0)
        sliders['x3_td'].set_val(1.5)
    
    preset_btn1.on_clicked(load_preset1)
    
    plt.show()
    return fig

# Run the interactive explorer
if __name__ == "__main__":
    create_isospectral_explorer_wide_range()
