from setuptools import setup, find_packages

setup(
    name='AUV_Project',
    version='0.1.0',
    description='in5490 project 4',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Jonas E. Wenberg, Sigmund, Thisan',
    author_email='jonaen@uio.no',
    url='https://github.com/ThisanK25/AUV_project', 
    packages=find_packages(),
    install_requires=[
        'numpy', 'xarray[complete]', 'scipy', 'matplotlib', 'pandas', 'packaging', 'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12', 
)