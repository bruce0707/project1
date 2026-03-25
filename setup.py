from setuptools import setup, find_packages,Setup
from typing import List
HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of requirements
    """
    with open(file_path) as f:
        requirement = f.readlines()
        requirement = [req.replace("\n","") for req in requirement]
        if HYPHEN_E_DOT in requirement:
            requirement.remove(HYPHEN_E_DOT)        
    return requirement    

setup(
    name="mlproject",
    version="0.1.0",
    author="Ankush Kumar",
    author_email="ankushhumai63@gmail.com",
    description="My first end to end machine learning project",
    packages=find_packages(),
    install_requires=["flask","numpy","seaborn","scikit-learn","pandas","xgboost"
    ],
)