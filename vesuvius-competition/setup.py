from setuptools import setup, find_packages

setup(
    name="vesuvius-competition",
    version="0.1.0",
    author="Your Name",
    description="Vesuvius Challenge Competition Solution",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#") and line.strip()
    ],
    entry_points={
        "console_scripts": [
            "train-vesuvius=src.training.train:main",
            "predict-vesuvius=src.inference.predict:main",
        ],
    },
)
