# near_inertial_wave_statistics_jfm2025
To reproduce the simulations described in the JFM paper, "Statistics of Near-Inertial Waves Over a Background Flow via Quantum and Statistical Mechanics,"

To run and analyze a simulation:
1) Filter the background flow field using the Jupyter notebook "filter_vortexGaz.ipynb" to generate the file "psi.npz." Note that the background flow field, "Flowfield_Meunier.mat," corresponds to an instantaneous snapshot of the 2D Navier-Stokes equations in the transient regime.
2) Run a simulation using the Python script "run_simu_torch.py." You can modify the parameters at the top of the script. (Note: This script uses the PyTorch library to parallelize array operations on a GPU.)
3) Concatenate the fields saved during the simulation using the "concatenate_fields.py" script.
4) Analyze the results using the Jupyter notebook "analysis_simu.ipynb."

The authors of the article can provide more information upon request.
