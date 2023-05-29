# Example (H<sub>2</sub> + Ca) 
<p>A typical PyQCAMS simulation folder for the reaction H<sub>2</sub> + Ca. The input file `inputs.json` contains all necessary parameters. Numerical potentials `h2_pec.dat` and `cah_au.txt` are in atomic units. The script `sim.py` runs trajectories in parallel, and loops over a list of impact parameters. There are both short (`results/_short.csv` and long (`results/_long.csv`) outputs of the results.</p>

<p>The `example.ipynb` Jupyter notebook is a tutorial notebook to guide the user through the steps and attributes of the PyQCAMS program, using the H<sub>2</sub> + Ca reaction as an example.</p>

<p>The `analysis.md` is an example usage of `pandas` to analyze a large dataset to create distribution and rate of formation plots. </p> 

