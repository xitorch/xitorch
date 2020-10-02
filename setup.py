import os
from setuptools import setup, find_packages

verfile = os.path.abspath(os.path.join("xitorch", "version.py"))
version = {"__file__": verfile}
with open(verfile, "r") as fp:
    exec(fp.read(), version)

setup(
    name='xitorch',
    version=version["get_version"](),
    description='Differentiable scientific computing library',
    url='https://xitorch.readthedocs.io/',
    author='mfkasim1',
    author_email='firman.kasim@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.8.2",
        "scipy>=1.1.0",
        "matplotlib>=1.5.3",
        "torch>=1.5",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: BSD License",

        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="project library linear-algebra autograd functionals",
    zip_safe=False
)
