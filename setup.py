from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="custom_yolo_lib",
    version="0.0.1",
    description="Object Detection with PyTorch",
    author="Emmanuel Vasilopoulos",
    author_email="emmanouilvasilopoulos@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
)
