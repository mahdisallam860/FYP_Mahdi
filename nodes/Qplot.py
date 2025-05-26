import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = "/home/alma/Downloads/reward_per_episode.xlsx"  # Adjust the path if needed
df = pd.read_excel(file_path)

# Ensure the dataframe has the expected columns
print(df.head())  # Debugging step to check data structure

# Assuming the file has columns like 'Episode' and 'Steps'
episode_col = df.columns[0]  # First column (Episode)
steps_col = df.columns[1]    # Second column (Steps)

episodes = df[episode_col]
steps = df[steps_col]

# Apply moving average for smoothing (ensuring it starts from the beginning)
window = 50  # Change window size as needed
moving_avg = np.convolve(steps, np.ones(window)/window, mode='same')  # Use 'same' to keep the length

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(episodes, steps, alpha=0.3, label="Raw Steps", color="blue")

plt.title("Reward per Episode Throughout Training")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(False)

# Save the plot
plt.savefig("steps_per_episode_plot.png", dpi=300)
plt.show()
