import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DominantSparseEigenAD",
    version="0.0.1",
    author="Hao Xie",
    author_email="xiehao18@iphy.ac.cn",
    description="Reverse-mode AD of dominant sparse eigensolver using Pytorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="automatic differentiation dominant eigensolver Pytorch",
    url="https://github.com/buwantaiji/DominantSparseEigenAD",
    packages=["DominantSparseEigenAD"],
    install_requires=[
            "numpy",
            "torch>=1.3.0",
        ],
    python_requires='>=3',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
