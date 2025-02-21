from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path:str) -> List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        return requirements

setup(
    name='Credit Card Fraud Detection',
    version='0.0.1',
    author='MD Nasim Akram',
    author_email='nasimakram6200@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)