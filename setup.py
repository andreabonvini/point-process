import json

from setuptools import find_packages, setup

tests_require = []


with open("Pipfile.lock") as fd:
    lock_data = json.load(fd)
    install_requires = [
        package_name + package_data["version"]
        for package_name, package_data in lock_data["default"].items()
    ]

setup(
    name="pointprocess",
    version="0.3",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "docs", "bin"]
    ),
    install_requires=install_requires,
    author="Andrea Bonvini",
    author_email="a.bonvini96@gmail.com",
)
