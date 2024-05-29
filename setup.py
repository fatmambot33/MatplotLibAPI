import os
from setuptools import setup, find_packages

# Function to read requirements.txt
def parse_requirements(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as file:
            lines = file.read().splitlines()
        # Filter out any empty lines or comments
        return [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return []

# Read the requirements from requirements.txt
requirements = parse_requirements('requirements.txt')

setup(
    name='MatplotLibAPI',
    version='v3.0.2',
    packages=find_packages(),
    install_requires=requirements,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
