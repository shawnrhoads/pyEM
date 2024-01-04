from setuptools import find_packages, setup

# Get the repository owner and name from the GitHub URL
github_url = 'https://github.com/shawnrhoads/pyEM'

# Create a list of author strings
authors = ["Shawn A. Rhoads"]

# Get long description
with open("README.md", "r") as fh:
    __long_description__ = fh.read()

# Get requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pyEM',
    version='v0.0.1',
    description='Python implementation of the Hierarchical Expectation Maximization algorithm with MAP estimation for fitting models to behavioral data',
    url=github_url,
    author=', '.join(authors), 
    packages=find_packages(),
    install_requires=required,
)