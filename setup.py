from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="xai4chem",
    author="Hellen Namulinda",
    author_email="hellennamulinda@gmail.com",
    description="Explainable AI for Chemistry", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ersilia-os/xai4chem",
    license="GPLv3",
    python_requires=">=3.7",
    packages=find_packages(exclude=("utilities")),
    classifiers=[  
        # 'Programming Language :: Python :: 3.7',
        # 'Programming Language :: Python :: 3.8',
        # 'Programming Language :: Python :: 3.9',
        # 'Programming Language :: Python :: 3 :: Only',
        # 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        # 'Operating System :: OS Independent',
        # 'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[""], 
    keywords="explainable-ai, chemistry, xai, machine learning, drug-discovery",
    project_urls={
        "Documentation": "",
        "Source Code": "https://github.com/ersilia-os/xai4chem",
    },
)