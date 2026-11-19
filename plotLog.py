from stable_baselines3.common import results_plotter
from matplotlib import pyplot as plt

# Plot the rewards over time
log_dir = './log/'
TIME_STEPS =   2_000_000_000
results_plotter.plot_results([log_dir], TIME_STEPS, results_plotter.X_TIMESTEPS, "Learning Curve")
plt.show()

while True:
    pass
