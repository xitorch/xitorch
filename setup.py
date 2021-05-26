import re
import os
from setuptools import setup, find_packages

module_name = "xitorch"
github_url = "https://github.com/xitorch/xitorch/tree/master/"
raw_github_url = "https://raw.githubusercontent.com/xitorch/xitorch/master/"

# open readme and convert all relative path to absolute path
with open("README.md", "r") as f:
    long_desc = f.read()

link_pattern = re.compile(r"\(([\w\-/]+)\)")
img_pattern  = re.compile(r"\(([\w\-/\.]+)\)")
link_repl = r"(%s\1)" % github_url
img_repl  = r"(%s\1)" % raw_github_url
long_desc = re.sub(link_pattern, link_repl, long_desc)
long_desc = re.sub(img_pattern, img_repl, long_desc)

file_dir = os.path.dirname(os.path.realpath(__file__))
absdir = lambda p: os.path.join(file_dir, p)

############### versioning ###############
verfile = os.path.abspath(os.path.join(module_name, "version.py"))
version = {"__file__": verfile}
with open(verfile, "r") as fp:
    exec(fp.read(), version)

############### setup ###############

def get_requirements(fname):
    with open(absdir(fname), "r") as f:
        return [line.strip() for line in f.read().split("\n") if line.strip() != ""]

build_version = "XITORCH_BUILD" in os.environ

setup(
    name=module_name,
    version=version["get_version"](build_version),
    description='Differentiable scientific computing library',
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url='https://xitorch.readthedocs.io/',
    author='mfkasim1',
    author_email='firman.kasim@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=get_requirements("requirements.txt"),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="project library linear-algebra autograd functionals",
    zip_safe=False
)
