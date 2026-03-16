from setuptools import setup, find_packages

setup(
    name="tda-mlops",
    version="0.4.0",
    description="Topological Data Analysis for Production-Grade Anomaly Detection",
    author="Kanha",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "ripser>=0.6",
        "persim>=0.3",
        "fastapi>=0.100",
        "uvicorn>=0.23",
        "mlflow>=2.0",
        "pydantic>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "httpx>=0.24",
        ]
    },
)