from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="causal-econ",
    version="0.1.0",
    author="Nikolay Voytov",
    author_email="not@today.com",
    description="A comprehensive library for causal inference in economic analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/causal-econ",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scipy>=1.6.0",
        "scikit-learn>=0.24.0",
        "torch>=1.8.0",
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
        "kaleido>=0.2.0",  # For plotly image export
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    include_package_data=True,
    package_data={
        "causal_econ": ["examples/data/*.csv"],
    },
)
