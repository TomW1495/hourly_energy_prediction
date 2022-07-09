from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = "\n".join(
    [line for line in long_description.split("\n") if not line.startswith("<img")]
)


setup(
    name="energy_prediction",
    description="Machine Learning experiments on energy predictions",
    author="Tom Watsham",
    author_email="tom.watsham@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomW1495/energy_prediction",
    version="0.0.1",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    packages=["energy_dataset"],
    install_requires=[
        # https://github.com/pytorch/pytorch/issues/78362
        "protobuf==3.20.1",
        "tqdm>=4.61.1",
        "torch>=1.11.0",
        "torchvision>=0.12.0",
        "kaggle",
        "sklearn"
    ],
    python_requires=">=3.8",
    include_package_data=True,
)