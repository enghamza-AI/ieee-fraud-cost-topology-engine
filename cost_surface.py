
# cost_surface.py


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D    
from typing import Tuple, Dict
import os



GRID_RESOLUTION = 30


N_THRESHOLDS = 50

OUTPUT_DIR = 'outputs'



def compute_cost_grid(
    y_test: pd.Series,
    y_prob: np.ndarray,
    fp_cost_range: Tuple[float, float] = (10, 500),
    fn_cost_range: Tuple[float, float] = (100, 5000),
    grid_resolution: int = GRID_RESOLUTION,
    verbose: bool = True
) -> Dict:
  

   
    if len(y_test) != len(y_prob):
        raise ValueError(
            f"y_test length ({len(y_test)}) != y_prob length ({len(y_prob)}). "
            "Make sure both come from the same test set."
        )
    if y_test.sum() == 0:
        raise ValueError("No fraud cases in y_test — cost calculation is meaningless.")

   
    fp_costs = np.linspace(fp_cost_range[0], fp_cost_range[1], grid_resolution)
    fn_costs = np.linspace(fn_cost_range[0], fn_cost_range[1], grid_resolution)

    thresholds = np.linspace(0.01, 0.99, N_THRESHOLDS)

    fp_counts = np.zeros(N_THRESHOLDS)   # false positives at each threshold
    fn_counts = np.zeros(N_THRESHOLDS)   # false negatives at each threshold

    if verbose:
        print(f"[1/3] Pre-computing FP/FN at {N_THRESHOLDS} thresholds...")

    for i, t in enumerate(thresholds):
        y_pred = (y_prob >= t).astype(int)
      
        fp_counts[i] = np.sum((y_pred == 1) & (y_test == 0))   # flagged legit
        fn_counts[i] = np.sum((y_pred == 0) & (y_test == 1))   # missed fraud

   
    min_cost_grid   = np.zeros((grid_resolution, grid_resolution))
    opt_thresh_grid = np.zeros((grid_resolution, grid_resolution))

    if verbose:
        print(f"[2/3] Computing {grid_resolution}×{grid_resolution} cost grid...")

    
    for i, fpc in enumerate(fp_costs):
        for j, fnc in enumerate(fn_costs):

           
            total_costs = fp_counts * fpc + fn_counts * fnc

            
            best_idx = np.argmin(total_costs)

            
            min_cost_grid[i, j]   = total_costs[best_idx]
            opt_thresh_grid[i, j] = thresholds[best_idx]

   
    FP_mesh, FN_mesh = np.meshgrid(fp_costs, fn_costs)
    
    cost_for_plot   = min_cost_grid.T
    thresh_for_plot = opt_thresh_grid.T

    if verbose:
        print(f"[3/3] Grid complete.")
        print(f"      Cost range   : ${min_cost_grid.min():,.0f} – ${min_cost_grid.max():,.0f}")
        print(f"      Threshold range: {opt_thresh_grid.min():.2f} – {opt_thresh_grid.max():.2f}")

    return {
        'fp_costs'        : fp_costs,
        'fn_costs'        : fn_costs,
        'thresholds'      : thresholds,
        'fp_counts'       : fp_counts,
        'fn_counts'       : fn_counts,
        'min_cost_grid'   : min_cost_grid,
        'opt_thresh_grid' : opt_thresh_grid,
        'FP_mesh'         : FP_mesh,
        'FN_mesh'         : FN_mesh,
        'cost_for_plot'   : cost_for_plot,
        'thresh_for_plot' : thresh_for_plot,
    }



def plot_cost_surface_static(
    grid_data: Dict,
    stakeholder_points: Dict = None,
    save_path: str = None,
    view_elev: float = 30,
    view_azim: float = 225
) -> plt.Figure:
  

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

   
    surf = ax.plot_surface(
        grid_data['FP_mesh'],
        grid_data['FN_mesh'],
        grid_data['cost_for_plot'],
        facecolors=plt.cm.RdYlGn_r(grid_data['thresh_for_plot']),  # reversed: low thresh = green
        alpha=0.85,
        rstride=1,   
        cstride=1,    
        linewidth=0,
        antialiased=True
    )

    
    if stakeholder_points:
        fp_costs   = grid_data['fp_costs']
        fn_costs   = grid_data['fn_costs']
        cost_grid  = grid_data['cost_for_plot']

        marker_colors = {'Bank': '#e74c3c', 'Merchant': '#3498db', 'Regulator': '#f39c12'}

        for name, (fpc, fnc) in stakeholder_points.items():
            
            i = np.argmin(np.abs(fp_costs - fpc))
            j = np.argmin(np.abs(fn_costs - fnc))
            z = cost_grid[j, i]

            color = marker_colors.get(name, 'purple')
            ax.scatter([fpc], [fnc], [z], color=color, s=120, zorder=10,
                       depthshade=False, edgecolors='white', linewidth=1.5)
            ax.text(fpc, fnc, z * 1.08, f' {name}', color=color, fontsize=9, fontweight='bold')

    
    ax.set_xlabel('FP Cost — False Alarm\n(blocking a real customer, $)',
                  fontsize=8, labelpad=10)
    ax.set_ylabel('FN Cost — Missed Fraud\n(fraud slips through, $)',
                  fontsize=8, labelpad=10)
    ax.set_zlabel('Minimum Total Business Loss ($)', fontsize=8, labelpad=10)
    ax.set_title(
        'Cost Surface — Optimal Total Loss by Cost Structure\n'
        'Colour = optimal decision threshold at that point',
        fontsize=11, pad=15
    )

    
    mappable = plt.cm.ScalarMappable(
        cmap='RdYlGn_r',
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Optimal Threshold', fontsize=8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0.0\n(aggressive)', '0.25', '0.5', '0.75', '1.0\n(conservative)'])

    
    ax.view_init(elev=view_elev, azim=view_azim)

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Static surface saved → {save_path}")

    return fig



def animate_cost_surface(
    grid_data: Dict,
    stakeholder_points: Dict = None,
    save_path: str = 'outputs/cost_surface.gif',
    n_frames: int = 72,
    fps: int = 18
) -> None:
   

    print(f"Building animation ({n_frames} frames at {fps} fps)...")
    print("This may take 30–90 seconds. Watch the frame counter.")

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')

    
    ax.plot_surface(
        grid_data['FP_mesh'],
        grid_data['FN_mesh'],
        grid_data['cost_for_plot'],
        facecolors=plt.cm.RdYlGn_r(grid_data['thresh_for_plot']),
        alpha=0.85, rstride=1, cstride=1,
        linewidth=0, antialiased=True
    )

   
    if stakeholder_points:
        fp_costs  = grid_data['fp_costs']
        fn_costs  = grid_data['fn_costs']
        cost_grid = grid_data['cost_for_plot']
        colors    = {'Bank': '#e74c3c', 'Merchant': '#3498db', 'Regulator': '#f39c12'}

        for name, (fpc, fnc) in stakeholder_points.items():
            i = np.argmin(np.abs(fp_costs - fpc))
            j = np.argmin(np.abs(fn_costs - fnc))
            z = cost_grid[j, i]
            ax.scatter([fpc], [fnc], [z], color=colors.get(name, 'purple'),
                       s=120, zorder=10, depthshade=False,
                       edgecolors='white', linewidth=1.5)
            ax.text(fpc, fnc, z * 1.08, f' {name}', color=colors.get(name, 'purple'),
                    fontsize=8, fontweight='bold')

    ax.set_xlabel('FP Cost ($)', fontsize=8)
    ax.set_ylabel('FN Cost ($)', fontsize=8)
    ax.set_zlabel('Min Total Loss ($)', fontsize=8)
    ax.set_title('Fraud Cost Surface — Rotating View', fontsize=11, pad=12)

    # Colorbar
    mappable = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(0, 1))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.4, pad=0.1)
    cbar.set_label('Optimal Threshold', fontsize=7)

   
    def update(frame):
        azimuth = (frame * (360 / n_frames)) % 360   
        elevation = 20 + 15 * np.sin(np.radians(frame * 2))   
        ax.view_init(elev=elevation, azim=azimuth)

        if frame % 10 == 0:
            print(f"  Frame {frame}/{n_frames}...")

        return fig,

    
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 // fps,
        blit=False
    )

   
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    anim.save(
        save_path,
        writer='pillow',
        fps=fps,
        dpi=100   
    )

    plt.close(fig)    
    print(f"Animation saved → {save_path}")
    print(f"File size tip: if it's >5MB, reduce n_frames or dpi.")



def find_optimal_threshold(
    y_test: pd.Series,
    y_prob: np.ndarray,
    fp_cost: float,
    fn_cost: float,
    n_thresholds: int = 200
) -> Tuple[float, float]:
   
    if fp_cost <= 0 or fn_cost <= 0:
        raise ValueError(
            f"Costs must be positive. Got fp_cost={fp_cost}, fn_cost={fn_cost}"
        )

    thresholds  = np.linspace(0.01, 0.99, n_thresholds)
    total_costs = np.zeros(n_thresholds)

    for i, t in enumerate(thresholds):
        y_pred = (y_prob >= t).astype(int)
        fp = np.sum((y_pred == 1) & (np.array(y_test) == 0))
        fn = np.sum((y_pred == 0) & (np.array(y_test) == 1))
        total_costs[i] = fp * fp_cost + fn * fn_cost

    best_idx = np.argmin(total_costs)
    return float(thresholds[best_idx]), float(total_costs[best_idx])



if __name__ == '__main__':
    print("cost_surface.py — standalone test with synthetic data\n")

    rng = np.random.default_rng(42)
    n = 3000

    y_test_fake = pd.Series((rng.random(n) < 0.035).astype(int))
    y_prob_fake = np.where(
        y_test_fake == 1,
        rng.beta(5, 2, n),
        rng.beta(2, 8, n)
    )

  
    stakeholders = {
        'Bank'      : (50,  2000),   
        'Merchant'  : (200, 300),    
        'Regulator' : (10,  5000),   
    }

   
    grid = compute_cost_grid(y_test_fake, y_prob_fake, verbose=True)

   
    fig = plot_cost_surface_static(
        grid,
        stakeholder_points=stakeholders,
        save_path='outputs/cost_surface_static.png'
    )

   
    print("\nOptimal thresholds per stakeholder:")
    for name, (fpc, fnc) in stakeholders.items():
        thresh, cost = find_optimal_threshold(y_test_fake, y_prob_fake, fpc, fnc)
        print(f"  {name:12s}: threshold = {thresh:.2f}  |  min cost = ${cost:,.0f}")

    
    animate_cost_surface(grid, stakeholder_points=stakeholders,
                         save_path='outputs/cost_surface.gif', n_frames=36, fps=12)

    plt.show()
    print("\nAll cost surface outputs generated.")