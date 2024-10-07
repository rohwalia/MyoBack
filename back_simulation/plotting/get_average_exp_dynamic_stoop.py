import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from scipy.ndimage import gaussian_filter1d

def process_time_series_separately(segments, folder_name):
    # Define a common time grid (adjust time range and step based on your data)
    common_time = np.linspace(0, 1, 1000)  # 0 to 1 second, 100 points for interpolation
    
    all_segments = []  # Store interpolated data for all segments across files

    # Iterate over each file (corresponding to the 9 subarrays)
    for i in range(1, 10):
        file_name = f'SUB{i}.csv'
        file_path = '/home/rwalia/MyoBack/back_simulation/plotting/Experiment_data/' + folder_name + "/" + file_name
        
        if not os.path.isfile(file_path):
            print(f"{file_path} does not exist.")
            continue

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, delimiter=';', decimal='.')
        
        if {'Column1', 'Column2', 'Column3'}.issubset(df.columns):
            time = np.array(df['Column1']) / 1000  # Convert ms to seconds
            column2 = np.array(df['Column2'])
            column3 = np.array(df['Column3'])
            
            # Process the segments for this file
            file_segments = segments[i - 1]  # The segment ranges for the current file
            
            for segment in file_segments:
                start, end = segment
                
                # Extract the data within the start and end times
                mask = (time >= start) & (time <= end)
                time_segment = time[mask]
                data_segment_col2 = column2[mask]
                data_segment_col3 = column3[mask]

                # Reset time for the segment
                time_segment_reset = time_segment - time_segment[0]
                
                # Normalize time to [0, 1] range for interpolation
                time_segment_normalized = time_segment_reset / time_segment_reset[-1]
                
                # Interpolate data for both Column2 and Column3 to match the common time grid
                interp_func_col2 = interp1d(time_segment_normalized, data_segment_col2, kind='linear', bounds_error=False, fill_value="extrapolate")
                interp_func_col3 = interp1d(time_segment_normalized, data_segment_col3, kind='linear', bounds_error=False, fill_value="extrapolate")
                
                interpolated_segment_col2 = interp_func_col2(common_time)
                interpolated_segment_col3 = interp_func_col3(common_time)

                #interpolated_segment_col2 = np.clip(interpolated_segment_col2, 0, None)
                #interpolated_segment_col3 = np.clip(interpolated_segment_col3, 0, None)

                # Append both interpolated segments (Column2 and Column3) to the list
                if np.mean(interpolated_segment_col2)<np.mean(interpolated_segment_col3):
                    all_segments.append(interpolated_segment_col2)
                else:
                    all_segments.append(interpolated_segment_col3)
                
        else:
            print(f"Missing required columns in {file_path}")
    
    # Convert the list of segments to a numpy array for further processing
    all_segments = np.array(all_segments)

    # Compute the mean and standard deviation across all segments
    mean_segments = np.mean(all_segments, axis=0)
    std_segments = np.std(all_segments, axis=0)
    return mean_segments, std_segments

# Example array with start and end times (your input array)
segments_lc = [
    [[1.822, 6.879], [8.763, 16.021], [17.936, 24.021], [26.771, 33.044], [35.75, 42.225]],
    [[1.64, 7.468], [10.425, 16.39], [19.277, 25.78], [28.391, 34.64], [38.622, 44.844]],
    [],
    [[8.873, 15.413], [16.694, 22.951], [24.354, 30.984], [31.328, 38.466], [39.108, 45.629]],
    [[3.708, 9.256], [11.303, 16.557], [18.657, 24.373], [26.218, 31.948], [33.697, 38.962]],
    [[2.203, 8.366], [11.637, 17.438], [20.317, 26.403], [29.039, 35.457], [38.318, 44.236]],
    [[4.132, 10.347], [13.127, 19.142], [22.147, 28.007], [31.047, 37.44], [39.908, 46.126]],
    [[4.63, 11.431], [13.862, 20.247], [21.87, 28.777], [31.9, 38.131], [40.697, 47.023]],
    [[2.69, 8.506], [10.492, 16.124], [17.773, 24.001], [25.596, 30.902], [32.764, 38.741]]
]

segments_mark = [
    [[1.822, 6.879], [8.763, 16.021], [17.936, 24.021], [26.771, 33.044], [35.75, 42.225]],
    [[1.64, 7.468], [10.425, 16.39], [19.277, 25.78], [28.391, 34.64], [38.622, 44.844]],
    [],
    [], #[[8.873, 15.413], [16.694, 22.951], [24.354, 30.984], [31.328, 38.466], [39.108, 45.629]],
    [], #[[3.708, 9.256], [11.303, 16.557], [18.657, 24.373], [26.218, 31.948], [33.697, 38.962]],
    [], #[[2.203, 8.366], [11.637, 17.438], [20.317, 26.403], [29.039, 35.457], [38.318, 44.236]],
    [], #[[4.132, 10.347], [13.127, 19.142], [22.147, 28.007], [31.047, 37.44], [39.908, 46.126]],
    [], #[[4.63, 11.431], [13.862, 20.247], [21.87, 28.777], [31.9, 38.131], [40.697, 47.023]],
    [] #[[2.69, 8.506], [10.492, 16.124], [17.773, 24.001], [25.596, 30.902], [32.764, 38.741]]
]

stiffness = 1527.51
sampling_rate = 100  # Assuming 100 Hz data frequency
time_step = 1 / sampling_rate  # Each time step is 0.01 seconds

def parse_column(column):
    """Parse column data into a 2D numpy array."""
    return np.array([np.fromstring(x.strip('[]'), sep=',') for x in column])

def compute_and_plot_distances(folder_name, segments):
    all_flex = []  # A single list to hold all interpolated distances (both RL and LL)
    
    # Iterate through CSV files in the folder
    for i in range(1, 10):
        file_name = f'SUB{i}.csv'
        file_path = '/home/rwalia/MyoBack/back_simulation/plotting/Experiment_data/'+folder_name+"/"+file_name
        print(file_path)
        
        if os.path.isfile(file_path):
            # Read the CSV file
            df = pd.read_csv(file_path, delimiter=';', decimal=',')
            
            # Parse required columns
            data_RL1 = parse_column(df['RL1'])
            data_RL2 = parse_column(df['RL2'])
            data_LL2 = parse_column(df['LL2'])
            data_LL1 = parse_column(df['LL1'])

            # Compute Euclidean distances
            dist_RL1_RL2 = np.sqrt(((data_RL1 - data_RL2) ** 2).sum(axis=1)) / 1000
            dist_LL2_LL1 = np.sqrt(((data_LL2 - data_LL1) ** 2).sum(axis=1)) / 1000

            # Adjust for the baseline (subtract the median of the first 100 points)
            dist_RL1_RL2 -= np.median(dist_RL1_RL2[:100], axis=0)
            dist_LL2_LL1 -= np.median(dist_LL2_LL1[:100], axis=0)

            # Apply stiffness to distances
            dist_RL1_RL2 *= stiffness
            dist_LL2_LL1 *= stiffness

            file_segments = segments[i - 1]  # Get the segment ranges for the current file

            # Process each segment for this file
            for segment in file_segments:
                start, end = segment
                start_idx = int(start * sampling_rate)  # Convert start time to index
                end_idx = int(end * sampling_rate)  # Convert end time to index

                # Extract the data within the segment
                dist_RL1_RL2_segment = dist_RL1_RL2[start_idx:end_idx]
                dist_LL2_LL1_segment = dist_LL2_LL1[start_idx:end_idx]

                # Reset time for the segment and normalize to a 0-1 range
                time_segment = np.arange(0, len(dist_RL1_RL2_segment)) * time_step
                time_segment_normalized = time_segment / time_segment[-1] if time_segment[-1] != 0 else time_segment
                
                # Interpolate the segment to match a common time grid (0 to 1 with 100 points)
                common_time = np.linspace(0, 1, 1000)
                dist_RL1_RL2_interp = np.interp(common_time, time_segment_normalized, dist_RL1_RL2_segment)
                dist_LL2_LL1_interp = np.interp(common_time, time_segment_normalized, dist_LL2_LL1_segment)

                #dist_RL1_RL2_interp = np.clip(dist_RL1_RL2_interp , 0, None)
                #dist_LL2_LL1_interp = np.clip(dist_LL2_LL1_interp , 0, None)

                if np.mean(dist_RL1_RL2_interp)>np.mean(dist_LL2_LL1_interp):
                    all_flex.append(dist_RL1_RL2_interp)
                else:
                    all_flex.append(dist_LL2_LL1_interp)
                
        else:
            print(f"File {file_path} does not exist.")
    
    # Convert the list to a numpy array for further processing
    all_flex = np.array(all_flex)
    print(all_flex)

    # Compute mean and standard deviation for the interpolated segments across all files
    mean_flex = np.mean(all_flex, axis=0)
    std_flex = np.std(all_flex, axis=0)
    return gaussian_filter1d(mean_flex, sigma=5), gaussian_filter1d(std_flex, sigma=5)

# Call the function with folder name
mean_segments_lc, std_segments_lc = process_time_series_separately(segments_lc, 'lc_aux_dynamic_stoop_0kg')
mean_segments_mark, std_segments_mark = compute_and_plot_distances('aux_dynamic_stoop_0kg', segments_mark)


common_time = np.linspace(0, 1, 1000) 
exo_forces = np.load("exo_forces_dynamic_stoop.npy")[:,0]

# Plot the result
plt.figure(figsize=(10, 6))

plt.plot(common_time, exo_forces, label='Simulated exoskeleton force', color='black')
plt.plot(common_time, mean_segments_mark, label='Experimental marker force', color='tab:red')
plt.fill_between(common_time, mean_segments_mark - std_segments_mark, mean_segments_mark + std_segments_mark, color='tab:red', alpha=0.3)
plt.plot(common_time, mean_segments_lc, label='Experimental loadcell force', color='tab:green')
plt.fill_between(common_time, mean_segments_lc - std_segments_lc, mean_segments_lc + std_segments_lc, color='tab:green', alpha=0.3)

plt.xlabel('Stoop Cycle')
plt.ylabel("Force [N]")
plt.legend()
plt.grid(True)
plt.savefig("dynamic_stoop.svg")