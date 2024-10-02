import opensim as osim

# Load the scaling tool using the XML setup file
scaling_tool = osim.ScaleTool('OpenSim_Python_Simulation/SETUP_Scale.xml')

# Perform scaling
scaling_tool.run()

# Load the newly scaled model
scaled_model = osim.Model('scaled_model.osim')

# Initialize the scaled model's state
state = scaled_model.initSystem()

# Print confirmation
print("Scaling complete. Scaled model saved as 'scaled_model.osim'")
