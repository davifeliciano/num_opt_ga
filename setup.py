import setuptools

with open("README.md", "r", encoding="utf-8") as rm:
    long_description = rm.read()

setuptools.setup(
    name="num_opt_ga",
    version="0.0.1",
    author="Davi Feliciano",
    author_email="dfeliciano37@gmail.com",
    description="An genetic algorithm to optmize real valued functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davifeliciano/num_opt_ga",
    project_urls={
        "Bug Tracker": "https://github.com/davi_feliciano/ising_model/issues",
        # "Documentation": "https://davifeliciano.github.io/num_opt_ga/index.html",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=["numpy", "matplotlib", "big_O"],
    python_requires=">=3.10",
)
