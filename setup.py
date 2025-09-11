from setuptools import find_packages, setup

# Get the repository owner and name from the GitHub URL
github_url = 'https://github.com/shawnrhoads/pyEM'

# Create a list of author strings
authors = ["Shawn A. Rhoads"]

# Get long description
with open("README.md", "r") as fh:
    __long_description__ = fh.read()

# Get requirements from requirements.txt, ignoring editable/local refs
with open('requirements.txt') as f:
    required = [line.strip() for line in f if line.strip() and not line.startswith('-e')]

setup(
    name='pyEM',
    version='v0.2.0',
    description=(
        'Python implementation of the Hierarchical Expectation Maximization '
        'algorithm with MAP estimation for fitting models to behavioral data'
    ),
    url=github_url,
    author=', '.join(authors), 
    packages=find_packages(),
    install_requires=required,
)
