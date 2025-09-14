import numpy as np
import matplotlib.pyplot as plt
import os

# --- THIS IS THE FILENAME WE WANT TO VISUALIZE ---
# This assumes you have run the clean data generation and renaming steps.
file_to_visualize = 'loc_testsprings5.npy'

# Construct the full path to the data file
file_path = os.path.join('data', file_to_visualize)

# --- CHECK IF THE FILE EXISTS BEFORE LOADING ---
if not os.path.exists(file_path):
    print(f"Error: The data file was not found at this location: {file_path}")
    print("Please make sure you have generated the 'springs5' dataset and renamed the files correctly.")
    exit()

# Load the specific test data
# The shape is (num_simulations, num_timesteps, dimensions, num_atoms)
loc_data = np.load(file_path)

# --- THIS IS THE CORRECT WAY TO GET THE NUMBER OF DRONES ---
# We get it from the LAST dimension of the full dataset.
num_drones = loc_data.shape[3]
print(f"Success! Loaded file '{file_to_visualize}' which contains trajectories for {num_drones} drones.")


# --- DATA PROCESSING FOR PLOTTING ---
# The data needs to be reshaped for matplotlib
# Reshape to: (num_simulations, num_drones, num_timesteps, dimensions)
loc_data_for_plot = np.transpose(loc_data, [0, 3, 1, 2])

# Let's pick the very first simulation to visualize
simulation_index = 0
simulation_data = loc_data_for_plot[simulation_index]


# --- PLOTTING ---
# Create a plot
plt.figure(figsize=(8, 8))

# Loop through each "drone" (or atom) in the simulation
for i in range(num_drones):
    # Get the trajectory for this one drone
    # trajectory[:, 0] is all the x-coordinates
    # trajectory[:, 1] is all the y-coordinates
    trajectory = simulation_data[i]
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Drone {i+1}')

# Add labels and a title
plt.title(f'Drone Swarm Trajectory (Simulation #{simulation_index} from {file_to_visualize})')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Show the plot
plt.show()