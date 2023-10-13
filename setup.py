from setuptools import setup, find_packages

setup(
    name='MatplotLibAPI',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib",
        "seaborn",
        "json",
        "logging",
        "scikit-learn"
    ],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
