import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nanograd",
    version="0.1",
    author="Daniel Sinkin",
    description="An even tinier scalar-valued autograd engine with a small PyTorch-like neural network library on top. Based on micrograd.",
    url="https://github.com/daniel-sinkin/nanograd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
