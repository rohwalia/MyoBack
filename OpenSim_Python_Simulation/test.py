import pandas as pd

def read_trc(file_path):
    # Read the TRC file, skipping the initial metadata lines (the first 6 lines)
    df = pd.read_csv(file_path, sep='\t', skiprows=4)  # Read two header rows
    return df

def get_markers(reference_trc_file):
    # Read the reference TRC file and extract the marker names from the first row of the header
    df_reference = read_trc(reference_trc_file)
    markers = df_reference.columns.get_level_values(0)[2:].tolist()  # Get all markers except Frame# and Time
    return markers

def keep_markers_based_on_another_trc(trc_file, reference_trc_file, output_file):
    # Read the TRC files
    df = read_trc(trc_file)
    df.dropna(axis=1, inplace=True)
    print(df.head)
    columns = pd.read_csv(trc_file, delim_whitespace=True, skiprows=3, nrows=1).columns.tolist()[2:]
    print(columns)

    columns_keep = pd.read_csv(reference_trc_file, delim_whitespace=True, skiprows=3, nrows=1).columns.tolist()[2:]
    print(columns_keep)

    indices = [i for i, item in enumerate(columns) if item not in columns_keep]
    print(indices)

    new_column = [item for i, item in enumerate(columns) if item in columns_keep]
    print(new_column)
    print(len(new_column))

    columns_to_drop = []
    for index in indices:
        columns_to_drop.extend([f'X{index+1}', f'Y{index+1}', f'Z{index+1}'])

    # Drop the specified columns
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore') 
    print(df)
    with open(output_file, 'w') as f:
        # Write the header as the first row separated by three tabs

        f.write('\t\t\t'.join(new_column) + '\n')

        # Write the DataFrame with tab separator

        df.to_csv(f, sep='\t', index=False, header=True)

# Example usage
trc_file = 'OpenSim_Python_Simulation/subject01_Cal01.trc'
reference_trc_file = 'OpenSim_Python_Simulation/output.trc'  # File with markers to keep
output_file = 'OpenSim_Python_Simulation/output_filtered.trc'

keep_markers_based_on_another_trc(trc_file, reference_trc_file, output_file)
