from setuptools import setup, find_packages

setup(
    name='vgsc',
    version='0.1.0',
    description='Video-based Gaussian Splat Coding framework',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author='Zhiwei Zhu',
    author_email='zhuzhiwei21@zju.edu.cn',
    url='https://github.com/zhuzhiwei99/VGSC',  
    license='MIT',
    packages=find_packages(exclude=["tests*", "__pycache__"]),
    include_package_data=True,
    install_requires=[
        'numpy',
        'torch',
        'plyfile',
        'tyro',
    ],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
