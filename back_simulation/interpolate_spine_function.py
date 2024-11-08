import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# x_name = '/jointset/L5_S1_IVDjnt/flex_extension/value'
# y_name = '/jointset/Abdjnt/Abs_r3/value'


# Read Excel file
# file_path = 'back_simulation/Values_Joinset_26_08.xlsx'  # Replace with your file path
file_path = 'back_simulation/AbdoPosition.xlsx'
df = pd.read_excel(file_path) #, usecols=[x_name, y_name])
# df = df.drop(df.index[::10])
# Extract x and y values
# x = df[x_name].values
# y = df[y_name].values

# x = df['Column1'][1:].astype(float).to_numpy()
# x = np.array([28.867944444444447, 43.895555555555546, 65.11938888888888])*(np.pi/180)
x = np.array([67.4334347826087, 82.573125, 95.49297826086956])*(np.pi/180)
index_closest_to_zero = (df['Column1'][1:].astype(float) - 0).abs().idxmin()
y = np.array([17.23, 23.063333333333333, 26.0925])*(np.pi/180)
# y = np.array([9.74425, 11.122222222222222, 22.0576875])*(np.pi/180)
# y = df['Column2'][1:].astype(float).to_numpy()
#y = y-y[index_closest_to_zero]

# Perform polynomial interpolation (degree 3 for this example)
degree = 2
coefs = np.polyfit(x, y, degree)
print(coefs)
poly = np.poly1d(coefs)

# Generate a smooth range of x values for plotting the polynomial
x_smooth = np.linspace(min(x), max(x), 500)
y_smooth = poly(x_smooth)

# Plot raw data points
plt.scatter(x, y, color='red', label='Raw Data')

# Plot polynomial interpolation
plt.plot(x_smooth, y_smooth, color='blue', label=f'Polynomial (degree {degree})')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Interpolation')
plt.legend()

# Show plot
plt.show()
