# PyQCAMS
Python Quasi-Classical Atom-Molecule Scattering, a Python package for atom-molecule scattering within the quasi-classical trajectory approach. 

## Installation
To install, navigate into the root directory of the repository and run
```python
pip install . 
``` 

## Usage
<p>The <code>example</code> folder contains a full example for simulating the reaction H<sub>2</sub> + Ca. 
An input dictionary containing atomic masses, interaction potentials, collision parameters, and integration parameters is used to run trajectories on atom-molecule systems. 
</p> 

## Output
<p> The collision energy is reported in Kelvin. All other attributes are in atomic units. The default long output for a tractory is: 
1. Initial state (vi, ji, E, b) representing initial vibrational state, vi, rotational state, ji, collision energy, E, and impact parameter, b.  
2. Product count (n12, n23, n31, nd, nc), where n is either 0 or 1. nij represents a bound molecule between atoms i and j, nd represents dissociation, and nc represents a three-atom intermediate complex at the end of the calculation.
3. Final state (v, vw, j, jw), where xw is the Gaussian weight of the state. For trajectories yielding nd = 1 or nc = 1, the final state outputs (0, 0, 0, 0).


## Analysis
We provide an <code>analysis</code> library with functions to calculate opacity, cross section, and rate coefficients of an outcome. They take the long output file as input. These have options for state-specific and non state-specific results.
The opacity function is a unitless probability, the cross section is in $`\mathrm{cm^2}`$, and the reaction rate coefficient is in $`\mathrm{cm^3/s}`$. 
</p>

## Data
A sample dataset to reproduce the figures in the example notebook can be found at https://figshare.com/s/8b923dab304005ae7a5c.  

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

