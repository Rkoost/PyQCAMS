# Example (H<sub>2</sub> + Ca) 
<p>A typical PyQCAMS simulation folder for the reaction H<sub>2</sub> + Ca. Two input files are needed `inputs.json` for calculation parameters and `input_v.py` for potential functions. Numerical potentials `h2_pec.dat` and `cah_au.txt` are in atomic units. These may be fit to the poly2 function in `pyqcams.vFactory`. The script `sim.py` runs trajectories in parallel, and loops over a list of impact parameters. There are both short (`results/_short.csv` and long (`results/_long.csv`) outputs of the results.</p>

<p>The `example.ipynb` Jupyter notebook is a tutorial notebook to guide the user through the steps and attributes of the PyQCAMS program, using the H<sub>2</sub> + Ca reaction as an example. The data files referenced as `cah_data/*.csv` are available at https://figshare.com/s/8b923dab304005ae7a5c.</p>

<p>The `analysis.md` is an example usage of `pandas` to analyze a large dataset to create distribution and rate of formation plots. </p> 

