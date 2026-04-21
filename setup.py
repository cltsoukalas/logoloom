from setuptools import setup, find_packages

setup(
    name="logoloom",
    version="0.1.0",
    description="Framework Alignment and Risk Overview System for Nuclear Reactor Design",
    author="Constantine Tsoukalas",
    author_email="cltsoukalas@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "sentence-transformers>=2.7.0",
        "scikit-learn>=1.4.0",
        "shap>=0.45.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "fastapi>=0.111.0",
        "streamlit>=1.35.0",
        "plotly>=5.22.0",
        "requests>=2.32.0",
        "joblib>=1.4.0",
    ],
)
