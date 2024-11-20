import numpy as np
from matplotlib import pyplot as plt
import os

save_dir = 'outputs/experiment3'

# Load data
loss_history = np.load(os.path.join(save_dir, 'loss_history.npy'))

# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history) + 1), loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.yscale('log')
plt.title("Training Loss History")
plt.legend()
plt.grid()

# Set y-axis ticks
y_ticks = [5e-2, 8e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 1e0]
plt.yticks(y_ticks, [f"{y:.0e}" for y in y_ticks])

# Save the plot
plt.savefig(os.path.join(save_dir, "loss_history.png"))
print("Loss history plot saved as loss_history.png")