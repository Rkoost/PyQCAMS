from setuptools import setup, find_packages

setup(
    name = 'PyQCAMS',
    version = '1.0.0',
    description = 'Python Quasi Classical Atom Molecule Scattering',
    author = 'Rian Koots',
    autor_email = 'rian.koots@stonybrook.edu',
    url = 'https://github.com/Rkoost/PyQCAMS',
    packages = find_packages(),
    install_requires =['ipython',
                       'matplotlib',
                       'numpy',
                       'pandas',
                       'scipy',
                       'multiprocess']
)