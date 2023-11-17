from setuptools import setup,find_packages
from typing import List


def get_require(file_path:str)->List[str]:

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements
setup(
    name='src',
    version='0.0.1',
    author='shivam',
    author_email='sg62795@gmail.com',
    packages=find_packages(),
    install_requires=get_require('requirements.txt')
)