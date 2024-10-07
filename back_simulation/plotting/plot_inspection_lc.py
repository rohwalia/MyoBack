import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_files(folder_name):
    # Iterate over CSV files from SUB1.csv to SUB9.csv
    for i in range(1, 10):
        file_name = f'SUB{i}.csv'
        file_path = '/home/rwalia/MyoBack/back_simulation/plotting/Experiment_data/'+folder_name+"/"+file_name
        
        if not os.path.isfile(file_path):
            print(f"{file_path} does not exist.")
            continue
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, delimiter=';', decimal=',')
        
        # Ensure the required columns exist in the DataFrame
        if {'Column1', 'Column2', 'Column3'}.issubset(df.columns):
            # Plot Column2 and Column3 against Column1
            plt.figure(figsize=(10, 6))
            plt.plot(df['Column1']/1000, df['Column2'], label='Loadcell 1', marker='o')
            plt.plot(df['Column1']/1000, df['Column3'], label='Loadcell 2', marker='x')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'Plot for {file_name}')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print(f"Missing required columns in {file_path}")


plot_csv_files('lc_aux_static_stoop_40')

# [[[1.933, 6.835], [10.282, 16.032], [18.757, 24.479], [28.166, 34.277], [36.807, 43.325]], [[2.22, 7.211], [10.281, 15.643], [19.503, 26.604], [30.923, 38.028], [40.413, 45.179]], [], [[5.55, 12.161], [14.781, 20.972], [22.492, 27.835], [29.626, 36.181], [38.707, 45.199]], [[3.589, 8.824], [11.18, 16.549], [18.446, 23.59], [25.789, 30.926], [33.26, 38.58]], [[2.994, 8.937], [11.877, 17.939], [20.916, 26.941], [29.84, 35.745], [38.72, 45.151]], [[4.077, 9.683], [12.986, 18.634], [22.228, 27.638], [31.092, 36.565], [39.904, 45.647]], [[4.514, 10.959], [13.449, 18.922], [22.603, 28.27], [31.596, 37.35], [40.713, 46.504]], [[2.219, 8.436], [10.369, 15.87], [18.078, 23.264], [25.452, 31.077], [32.972, 38.381]]]

# [[[1.822, 6.879], [8.763, 16.021], [17.936, 24.021], [26.771, 33.044], [35.75, 42.225]], [[1.64, 7.468], [10.425, 16.39], [19.277, 2.578], [28.391, 34.64], [38.622, 44.844]], [], [[8.873, 15.413], [16.694, 22.951], [24.354, 30.984], [31.328, 38.466], [39.108, 45.629]], [[3.708, 9.256], [11.303, 16.557], [18.657, 24.373], [26.218, 31.948], [33.697, 38.962]], [[2.203, 8.366], [11.637, 17.438], [20.317, 26.403], [29.039, 35.457], [38.318, 44.236]], [[4.132, 10.347], [13.127, 19.142], [22.147, 28.007], [31.047, 37.44], [39.908, 46.126]], [[4.63, 11.431], [13.862, 20.247], [21.87, 28.777], [31.9, 38.131], [40.697, 47.023]], [[2.69, 8.506], [10.492, 16.124], [17.773, 24.001], [25.596, 30.902], [32.764, 38.741]]]