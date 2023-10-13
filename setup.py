from setuptools import setup, find_packages

setup(
    name='MatplotLibAPI',
    version='v2.0.2',
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn"
    ],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
