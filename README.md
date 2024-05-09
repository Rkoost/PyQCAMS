# PyQCAMS
Python Quasi-Classical Atom-Molecule Scattering, a Python package for atom-molecule scattering within the quasi-classical trajectory approach. 

## Installation
To install, navigate into the root directory of the repository and run
```python
pip install . 
``` 

## Usage
<p>The <code>example</code> folder contains a full example for simulating the reaction H<sub>2</sub> + Ca. To run a trajectory, an input dictionary is required containing the following keywords:

1. m1, m2, m3: Atomic masses (a.u.)
2. E0: Collision energy (K)
3. b0: Impact parameter (Bohr)
4. R0: Initial distance (Bohr)
5. seed: Random number generator seed (default: None)
6. mol_12, mol_23, mol_31: Molecule class objects. mol_12 assumed to be initial bound molecule.
7. Vt, dVtdr12, dVtdr23, dVtdr31: Three-body interaction term (Hartree) and associated partial derivatives.
8. integ[t_stop, r_stop, r_tol, a_tol, econs, lcons]: Integration parameters. Respectively, stopping condition for time as a multiple of collision timescale, stopping condition for distance as a multiple of initial distance, relative and absolute error tolerance for integrator, energy and momentum conservation in atomic units.

The <code>Molecule</code> class is used to generate the three possible molecules, where the two-particle interactions are defined. Create three molecules to place in the input dictionary. For example, an H$_2$ molecule is generated:
```python
m1 = 1.008*constants.u2me
m2 = 1.008*constants.u2me
v12, dv12 = potentials.morse(de = 0.16, re = 1.4, alpha = 1.06)
mol12 = qct.Molecule(mi = m1, mj = m2, Vij = v12, dVij = dv12, vi = 0, ji = 0,
                     xmin = .5, xmax = 30, npts=1000)
``` 
where `mi,mj` are the masses (a.u.), `vi,ji` represent the initial rovibrational state, and `Vij, dVij` represent the two-body interaction and its derivative. `xmax, xmin` represent the region of interaction, and `npts` is the number of grid points used in the DVR to calculate the energy spectrum. To bypass the DVR step, the internal energy can be added using the argument `Ei`. 
</p> 

## Output
<p> The collision energy is reported in Kelvin. All other attributes are in atomic units. The default long output for a tractory is:

1. Initial state (vi, ji, E, b) representing initial vibrational state, vi, rotational state, ji, collision energy, E, and impact parameter, b. 
2. Product count (n12, n23, n31, nd, nc), where n is either 0 or 1. nij represents a bound molecule between atoms i and j, nd represents dissociation, and nc represents a three-atom intermediate complex at the end of the calculation.
3. Final state (v, vw, j, jw), where xw is the Gaussian weight of the state. For trajectories yielding nd = 1 or nc = 1, the final state outputs (0, 0, 0, 0). 

 </p>

## Analysis
<p> We provide an <code>analysis</code> library with functions to calculate opacity, cross section, and rate coefficients of an outcome. They take the long output file as input. These have options for state-specific and non state-specific results.
The opacity function is a unitless probability, the cross section is in $\mathrm{cm^2}$, and the reaction rate coefficient is in $\mathrm{cm^3/s}$. 
</p>

## Data
A sample dataset to reproduce the figures in the example notebook can be found at https://figshare.com/s/8b923dab304005ae7a5c.  

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

