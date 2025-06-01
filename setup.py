from setuptools import find_packages, setup

setup(
    name="clipsai",
    py_modules=["clipsai"],
    version="0.2.1",
    description=(
        "Clips AI is an open-source Python library that automatically converts long "
        "videos into clips"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Benjamin Smidt, Johann Ramirez, Armel Talla",
    author_email="support@clipsai.com",
    url="https://clipsai.com/",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "av~=10.0",
        "facenet-pytorch~=2.5",
        "matplotlib~=3.7",
        "mediapipe~=0.10",
        "nltk~=3.8",
        "numpy>=1.23,<2.0",
        "opencv-python~=4.8",
        "pandas>=1.5,<2.1",
        "psutil~=5.9",
        "pyannote.audio>=3.0,<3.2",
        "pyannote.core~=5.0",
        "pynvml~=11.5",
        "python-magic~=0.4",
        "scenedetect~=0.6",
        "scikit-learn~=1.3",
        "sentence-transformers~=2.2",
        "scipy>=1.10,<2.0",
        "torch>=1.13,<2.1",
        "whisperx @ git+https://github.com/m-bain/whisperx.git",
    ],
    zip_safe=False,
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://docs.clipsai.com/",
        "Homepage": "https://clipsai.com/",
        "Repository": "https://github.com/ClipsAI/clipsai",
        "Issues": "https://github.com/ClipsAI/clipsai/issues",
    },
    include_package_data=True,
    extras_require={
        "dev": [
            "black",
            "black[jupyter]",
            "build",
            "flake8",
            "ipykernel",
            "pytest",
            "twine",
        ],
    },
)
