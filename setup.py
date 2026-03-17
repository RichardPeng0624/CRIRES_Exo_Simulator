from setuptools import setup

setup(
    name='exocrires',
    version='1.14.1',
    # setup.py lives inside the package directory itself.
    # Explicitly map package name to current directory so editable/wheel installs work.
    packages=['exocrires'],
    package_dir={'exocrires': '.'},
    install_requires=[],
)# List your dependencies here
