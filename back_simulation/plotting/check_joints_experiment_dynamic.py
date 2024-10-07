import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

stiffness = 1527.51
sampling_rate = 100  # 100 Hz
time_step = 1 / sampling_rate  # Each time step is 0.01 seconds

def parse_column(column):
    """Parse column data into a 2D numpy array."""
    return np.array([np.fromstring(x.strip('[]'), sep=',') for x in column])

def compute_and_plot_distances(folder_name, segments):
    all_flex = []  # Single list for both right and left flexion
    
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

            # Calculate vectors for flexion extension
            vect_flex_r = data_LL2 - data_RL1
            vect_flex_l = data_RL2 - data_LL1

            downward_vector = np.array([0, -1, 0])
            flex_right = np.arccos(np.dot(vect_flex_r, downward_vector) / (np.linalg.norm(vect_flex_r, axis=1) * np.linalg.norm(downward_vector)))
            flex_left = np.arccos(np.dot(vect_flex_l, downward_vector) / (np.linalg.norm(vect_flex_l, axis=1) * np.linalg.norm(downward_vector)))

            # Adjust for flexion angle by removing the baseline (median of the first 100 points)
            flex_right -= np.median(flex_right[:100], axis=0)
            flex_left -= np.median(flex_left[:100], axis=0)

            file_segments = segments[i - 1]  # The segment ranges for the current file
            
            # Process each segment for this file
            for segment in file_segments:
                start, end = segment
                start_idx = int(start * sampling_rate)  # Convert start time to index
                end_idx = int(end * sampling_rate)  # Convert end time to index

                # Extract the data within the segment
                flex_right_segment = flex_right[start_idx:end_idx]
                flex_left_segment = flex_left[start_idx:end_idx]

                # Reset time for the segment and normalize to 0-1 range
                time_segment = np.arange(0, len(flex_right_segment)) * time_step
                time_segment_normalized = time_segment / time_segment[-1] if time_segment[-1] != 0 else time_segment
                
                # Interpolate the segment to match a common time grid (0 to 1 with 100 points)
                common_time = np.linspace(0, 1, 1000)
                flex_right_interp = np.interp(common_time, time_segment_normalized, flex_right_segment)
                flex_left_interp = np.interp(common_time, time_segment_normalized, flex_left_segment)

                # Append both right and left interpolated data to the same list
                if np.mean(flex_right_interp)>np.mean(flex_left_interp):
                    all_flex.append(flex_right_interp)
                else:
                    all_flex.append(flex_left_interp)
                
        else:
            print(f"File {file_path} does not exist.")
    
    # Convert the list to a numpy array for further processing
    all_flex = np.array(all_flex)

    # Compute mean and standard deviation for the segments across all files
    mean_flex = np.mean(all_flex, axis=0)
    print(mean_flex)
    std_flex = np.std(all_flex, axis=0)

    np.save("exo_joint_dynamic_stoop.npy", mean_flex)

    # Plot mean and standard deviation as shaded regions
    plt.figure(figsize=(12, 6))
    
    # Plot flexion mean with standard deviation
    plt.plot(common_time, np.degrees(mean_flex), label='Mean Flexion Extension (Right + Left)', color='purple')
    plt.fill_between(common_time, np.degrees(mean_flex - std_flex), np.degrees(mean_flex + std_flex), color='purple', alpha=0.3)
    
    plt.xlabel('Normalized Time (0-1)')
    plt.ylabel('Degrees')
    plt.title('Mean and Standard Deviation of Flexion Extension (Right + Left)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example array with start and end times (your input array)
segments_squat = [
    [[1.933, 6.835], [10.282, 16.032], [18.757, 24.479], [28.166, 34.277], [36.807, 43.325]],
    [], #[[2.22, 7.211], [10.281, 15.643], [19.503, 26.604], [30.923, 38.028], [40.413, 45.179]],
    [],
    [[5.55, 12.161], [14.781, 20.972], [22.492, 27.835], [29.626, 36.181], [38.707, 45.199]],
    [[3.589, 8.824], [11.18, 16.549], [18.446, 23.59], [25.789, 30.926], [33.26, 38.58]],
    [[2.994, 8.937], [11.877, 17.939], [20.916, 26.941], [29.84, 35.745], [38.72, 45.151]],
    [[4.077, 9.683], [12.986, 18.634], [22.228, 27.638], [31.092, 36.565], [39.904, 45.647]],
    [], #[[4.514, 10.959], [13.449, 18.922], [22.603, 28.27], [31.596, 37.35], [40.713, 46.504]],
    [] #[[2.219, 8.436], [10.369, 15.87], [18.078, 23.264], [25.452, 31.077], [32.972, 38.381]]
]

segments_stoop = [
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

# Example usage
compute_and_plot_distances('aux_dynamic_stoop_0kg', segments_stoop)
