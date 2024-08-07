import setuptools

with open("README.md", "r") as readme:
    long_description = readme.read()

with open("requirements.txt", "r") as req:
    requirements = req.read().splitlines()

setuptools.setup(
    name="higher_level_nav",
    version="0.1.0",
    author="DML group",
    author_email="",
    description=" D navigation at the highest level of abstraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 4 - Beta"
    ],
    python_requires=">=3.8",
    install_requires=requirements,

)