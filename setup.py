from distutils.core import setup

setup(
    name="scPredictor",
    version="0.1",
    packages=[
        "scPredictor",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "fastcluster",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "jupyterlab",
        "ipykernel",
        "h5py"
    ]
)




