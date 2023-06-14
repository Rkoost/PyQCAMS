from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory/"README.md").read_text()

setup(
    name = 'PyQCAMS',
    version = '1.0.3',
    description = 'Python Quasi Classical Atom Molecule Scattering',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author = 'Rian Koots',
    author_email = 'rian.koots@stonybrook.edu',
    url = 'https://github.com/Rkoost/PyQCAMS',
    packages = find_packages(),
    install_requires =[
                       'ipython>=7.30',
                       'matplotlib>=3.7.1',
                       'numpy>= 1.23.5' ,
                       'pandas>=1.2.4',
                       'scipy>=1.10.0',
                       'multiprocess>=0.70.14']
)
