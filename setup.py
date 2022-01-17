from setuptools import setup, find_packages

setup(
    name='kempers01',
    version='0.1.0',
    packages=find_packages(include=['Kempers01', 'Kempers01.*']),
    install_requires=[
        'numpy',
        'pyctp',
        'pandas',
        'kineticgas'
    ]
)