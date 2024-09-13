import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/rwalia/MyoBack/back_simulation/plotting')
import get_average_exp_static_loadcell 
import get_average_exp_static_marker
import math
import cv2

def plot_exo_forces(joint):
    # Charger les forces enregistrées
    exo_forces = np.load("exo_forces_static_stoop.npy")

    # Extraire les forces des deux actionneurs
    forces_actuator_1 = exo_forces[:, 0]

    exp_values_40 = [44.12, 40.726, 43.153, 31.731, 37.285000000000004, 40.311, 23.678000000000004, 22.455000000000002, 17.177000000000003, 17.771, 19.547, 20.639000000000003, 22.83, 20.979000000000003, 25.739, 33.12500000000001, 35.604000000000006, 34.83200000000001, 26.915000000000003, 36.11300000000001, 34.71900000000001, 39.576, 35.82599999999999, 41.665000000000006, 28.275999999999996, 32.002, 34.301, 43.42900000000001, 37.79100000000001, 32.648, 24.752000000000002, 23.844, 17.614000000000004, 25.169999999999998, 27.274, 29.347, 20.843, 18.654, 21.128999999999998, 31.17, 32.286, 33.904, 22.122, 31.657999999999998, 27.962]
    exp_values_60 = [65.06400000000001, 70.737, 67.95000000000002, 46.412000000000006, 50.045, 50.175000000000004, 30.906999999999993, 24.564, 29.873000000000005, 26.511000000000003, 23.821, 28.942000000000004, 33.114000000000004, 35.597, 27.865000000000006, 33.486000000000004, 37.43499999999999, 31.900999999999996, 33.12500000000001, 35.604000000000006, 34.83200000000001, 44.690999999999995, 44.37899999999999, 39.081, 63.98700000000001, 60.40999999999999, 64.17299999999999, 66.77000000000001, 68.387, 64.003, 44.51499999999999, 49.314, 51.159000000000006, 56.58, 49.089, 51.543, 30.922, 25.905999999999995, 31.927000000000003, 47.760000000000005, 47.95499999999999, 38.087, 35.746, 36.196999999999996, 36.506, 31.17, 32.286, 33.904, 35.537, 41.11600000000001, 39.07300000000001, 67.92200000000001, 64.091, 66.976]
    exp_values_80 = [69.69300000000001, 61.259, 54.02700000000001, 65.691, 60.510999999999996, 66.41600000000001, 38.311, 40.39600000000001, 39.592, 48.059, 49.404999999999994, 46.550999999999995, 62.63300000000001, 64.332, 53.23500000000001, 40.543, 42.27799999999999, 38.074000000000005, 37.604, 36.82899999999999, 36.865, 53.736999999999995, 55.18299999999999, 54.636, 67.378, 55.37799999999999, 57.808, 65.894, 59.09499999999999, 55.706, 57.622, 55.922999999999995, 61.382, 72.51199999999999, 71.113, 68.423, 43.507, 48.086000000000006, 42.526999999999994, 86.683, 86.5, 74.50300000000001, 42.789, 42.95099999999999, 38.535000000000004, 33.55400000000001, 32.011, 31.599000000000004, 61.78499999999999, 63.50200000000001, 65.619, 69.462, 59.348000000000006, 60.577000000000005]
    angle_values_40 = [34.684, 40.663, 26.933, 29.386, 30.152, 27.323, 25.543, 30.734, 31.806, 32.818, 25.451, 29.767, 25.255]
    angle_values_60 = [56.211, 46.680, 47.335, 46.669, 45.134, 45.94, 47.41, 41.689, 41.530]
    angle_values_80 = [69.7, 69.33, 70.82, 63.962, 68.56, 68.1, 59.78, 63.78, 67.69, 67.82, 64.427, 68.21, 65.61, 59.54]
    exp_values_40_m = [37.33801602248917, 30.367924215054945, 31.725199384141657, 34.803438168731894, 24.162110322487248, 25.583096604843746]
    exp_values_60_m = [31.586926001264025, 35.03528817980574, 31.73895412878165, 41.122838767422, 37.924774417048404, 41.744264814811764, 61.628360223764126, 71.72233965697188, 66.9178289666422, 30.227229463909794, 33.28194620695953, 34.72663330588501, 43.630900585268535, 39.11102834080059, 45.744375560649495]
    exp_values_80_m = [40.04439291418832, 41.492021613695215, 57.43772616109635, 58.498609844447465, 49.69476469903161, 48.169484590071995, 73.74511184849247, 65.42188341391254, 46.162397796535174, 53.91526876015231, 46.57324879839999, 53.14453089078582, 41.721676633477905, 88.58855313559151, 77.25065091716394, 62.1525392638061, 47.31373952941617, 49.68626214350717]

    std40 = np.std(exp_values_40)
    mean40= np.mean(exp_values_40)
    std60 = np.std(exp_values_60)
    mean60= np.mean(exp_values_60)
    std80 = np.std(exp_values_80)
    mean80= np.mean(exp_values_80)
    means =[mean40,mean60,mean80]
    stds =[std40,std60,std80]

    std40_m = np.std(exp_values_40_m)
    mean40_m= np.mean(exp_values_40_m)
    std60_m = np.std(exp_values_60_m)
    mean60_m= np.mean(exp_values_60_m)
    std80_m = np.std(exp_values_80_m)
    mean80_m= np.mean(exp_values_80_m)
    means_m=[mean40_m,mean60_m,mean80_m]
    stds_m=[std40_m,std60_m,std80_m]

    std40_angle = np.std(angle_values_40)
    mean40_angle = np.mean(angle_values_40)
    std60_angle = np.std(angle_values_60)
    mean60_angle = np.mean(angle_values_60)
    std80_angle = np.std(angle_values_80)
    mean80_angle = np.mean(angle_values_80)
    means_angle = [mean40_angle,mean60_angle,mean80_angle]
    stds_angle = [std40_angle,std60_angle,std80_angle]
    print(means_angle)
    res=500
    # Créer un vecteur pour l'axe des x (index des étapes de simulation)
    x = (np.linspace(-1.222, -0.314159, res)[::-1])*(180/np.pi)

    # Tracer les forces
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 100)  # Limites de l'axe y de 0 à 100
    plt.plot(-x, forces_actuator_1, label='Simulated Exoskeleton force', color='black')
    plt.errorbar(means_angle, means, xerr=stds_angle, yerr=stds, fmt='-o', label='Experimental loadcell force', color='tab:green')
    plt.errorbar(means_angle, means_m, xerr=stds_angle, yerr=stds_m, fmt='-o', label='Experimental marker force', color='tab:red')

    # Ajouter des titres et des légendes
    plt.xlabel("Flex extension [°]", fontdict={'fontname': 'Times New Roman'})
    plt.ylabel("Force [N]", fontdict={'fontname': 'Times New Roman'})
    plt.legend()
    plt.grid(True)
    
    # Afficher le graphique
    plt.savefig("static_stoop.svg")

if __name__ == '__main__':
    plot_exo_forces(joint="flex_extension")