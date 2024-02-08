from setuptools import setup, find_packages

with open("package_description.md", "r") as fh:
    long_description = fh.read()

setup(
    name="re_technical_report",
    version="0.0.6a1",
    author="Sebastian Cacean, Andreas Freivogel",
    author_email="sebastian.cacean@kit.edu, andreas.freivogel@unibe.ch",
    description="Helper functions for the report 'Assessing a Formal Model of Reflective Equilibrium'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/re-models/re-technical-report",
    packages=find_packages(),
    package_dir={'re_technical_report': 're_technical_report'},
    classifiers=["Programming Language :: Python :: 3.9",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
    python_requires='>=3.9',
    install_requires=['numpy', 'pandas','scipy', 'rethon'
                     ],
)
