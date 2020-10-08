import os
from setuptools import setup, find_packages
from distutils.command.build_py import build_py

try:
    import torch
except ImportError:
    raise ImportError("Please install torch before installing xitorch")

module_name = "xitorch"
file_dir = os.path.dirname(os.path.realpath(__file__))
absdir = lambda p: os.path.join(file_dir, p)


############### special functions template writing ###############
def sp_write_template():
    sp_template_file = absdir("xitorch/_impls/special/write_functions.py")
    sp_template_module = {"__file__": sp_template_file}
    with open(sp_template_file, "r") as fp:
        exec(fp.read(), sp_template_module)
    sp_template_module["main"]()

############### special functions compilation ###############
sp_ext_name = "_%s_special_impl" % module_name
sp_incl_dirs = [absdir("xitorch/_impls")]
sp_source_dirs = [
    absdir("xitorch/_impls/special/generated/")
]
sp_source_exts = [".cpp", ".c"]

def sp_get_sources():
    def _is_correct_ext(fname):
        for ext in sp_source_exts:
            if fname.endswith(ext):
                return True
        return False
    allfiles = []
    for sp_dir in sp_source_dirs:
        allfiles = allfiles + [os.path.join(sp_dir, f) for f in os.listdir(sp_dir) if _is_correct_ext(f)]
    print("Compiling:")
    print(allfiles)
    return allfiles

sp_sources = [
    absdir("xitorch/_impls/special/generated/batchfuncs.cpp"),
    absdir("xitorch/_impls/special/generated/bind.cpp"),
]

def get_torch_cpp_extension():
    from torch.utils.cpp_extension import CppExtension
    return CppExtension(
        name=sp_ext_name,
        sources=sp_get_sources(),
        include_dirs=sp_incl_dirs,
        # extra_compile_args=['-g'],
    )

def get_build_extension():
    from torch.utils.cpp_extension import BuildExtension
    class CustomBuildExtension(BuildExtension):
        def run(self):
            sp_write_template()
            BuildExtension.run(self)
    return CustomBuildExtension

############### versioning ###############
verfile = os.path.abspath(os.path.join(module_name, "version.py"))
version = {"__file__": verfile}
with open(verfile, "r") as fp:
    exec(fp.read(), version)

############### setup ###############

def get_requirements(fname):
    with open(absdir(fname), "r") as f:
        return [line.strip() for line in f.read().split("\n") if line.strip() != ""]

setup(
    name=module_name,
    version=version["get_version"](),
    description='Differentiable scientific computing library',
    url='https://xitorch.readthedocs.io/',
    author='mfkasim1',
    author_email='firman.kasim@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=get_requirements("requirements.txt"),
    setup_requires=get_requirements("setup-requirements.txt"),
    ext_modules=[get_torch_cpp_extension()],
    cmdclass={
        'build_ext': get_build_extension(),
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="project library linear-algebra autograd functionals",
    zip_safe=False
)
