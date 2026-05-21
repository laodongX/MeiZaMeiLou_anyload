from setuptools import setup,find_packages
with open(file="README.md",mode='r',encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="MeiZaMeiLou_anyload",
    version="0.1.0",
    author="KaMenRiDon",
    description="Universal, multi-modal, config-driven data loading pipeline for LLM/VLM training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/laodongX/MeiZaMeiLou_anyload",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "torchaudio>=2.0.0",
        "tqdm>=4.65.0",
    ],
)
