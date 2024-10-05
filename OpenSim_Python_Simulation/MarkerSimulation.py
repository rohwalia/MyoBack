import opensim as osim
import faulthandler
import sys

# Enable fault handler to print tracebacks on segfault
faulthandler.enable()

# Redirect stdout to a log file to capture OpenSim's output
log_file = open('opensim_log.txt', 'w')
sys.stdout = log_file

try:
    # Load the scaling tool using the XML setup file
    scaling_tool = osim.ScaleTool('OpenSim_Python_Simulation/subjectMyoBack_scaleSet.xml')

    # Perform scaling
    scaling_tool.run()

    # Load the newly scaled model
    scaled_model = osim.Model('scaled_model.osim')
    state = scaled_model.initSystem()

    print("Scaling complete. Scaled model loaded as 'scaled_model.osim'")
except Exception as e:
    print(f"An error occurred: {e}")

sys.stdout = sys.__stdout__
log_file.close()

