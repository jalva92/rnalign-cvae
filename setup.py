from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cvae-gene-expression",
    version="0.1.0",
    author="Jacob Alvarez",
    author_email="jalvarez@gis.a-star.edu.sg",
    description="A modular framework for training and evaluating conditional Variational Autoencoders (cVAE) on gene expression data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rnalign/cvae-gene-expression",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch==2.2.2",
        "pyro-ppl==1.9.0",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scikit-learn==1.4.2",
        "tqdm==4.66.2",
        "pyyaml==6.0.1",
        "jupyter==1.0.0",
        "notebook==7.1.3",
        "ipykernel==6.29.4",
        "matplotlib==3.8.4",
        "seaborn==0.13.2",
        "pytest==8.1.1",
        "black==24.3.0",
        "flake8==7.0.0",
        "mypy==1.9.0",
        "shap==0.45.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)