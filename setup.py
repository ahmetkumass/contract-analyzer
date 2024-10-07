from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="contract-analyzer",
    version="0.1.5",
    author="Ahmet Kumas",
    author_email="ahmetkumas@outlook.com",
    description="A RAG system for contract analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahmetkumass/contract-analyzer.git",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "transformers",
        "torch",
        "pdfplumber",
        "PyPDF2",
    ],
)