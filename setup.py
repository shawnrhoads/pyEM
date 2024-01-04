from setuptools import find_packages, setup
import requests

# Get the repository owner and name from the GitHub URL
github_url = 'https://github.com/shawnrhoads/pyEM'
owner, repo = github_url.split('/')[-2:]

# Get the list of contributors from the GitHub API
response = requests.get(f'https://api.github.com/repos/{owner}/{repo}/contributors')
contributors = response.json()

# Create a list of author strings in the format "Name <email>"
authors = [f"{c['login']}" for c in contributors]

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