# Example (H<sub>2</sub> + Ca) 
<p>A typical PyQCAMS simulation folder for the reaction H<sub>2</sub> + Ca. Two input files are needed <code>inputs.json</code> for calculation parameters and <code>input_v.py</code> for potential functions. Numerical potentials <code>h2_pec.dat</code> and <code>cah_au.txt</code> are in atomic units. These may be fit to the poly2 function in <code>pyqcams.vFactory</code>. The script <code>sim.py</code> runs trajectories in parallel, and loops over a list of impact parameters. There are both short (<code>results/_short.csv</code>) and long (<code>results/_long.csv</code>) outputs of the results.</p>

<p>The <code>example.ipynb</code> Jupyter notebook is a tutorial notebook to guide the user through the steps and attributes of the PyQCAMS program, using the H<sub>2</sub> + Ca reaction as an example. The data files referenced as <code>cah_data/*.csv</code> are available at https://figshare.com/s/8b923dab304005ae7a5c.</p>

<p>The <code>analysis.md</code> is an example usage of <code>pandas</code> to analyze a large dataset to create distribution and rate of formation plots. </p> 

