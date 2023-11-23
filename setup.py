from setuptools import setup, find_packages

setup(
    name='twobit_pytorchkit',
    version='0.1.0',
    author='Jeremy Berry',
    author_email='twobit.five@outlook.com',
    description='PyTorch classes to assist in training PyTorch models, plotting training metrics, and evaluation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/twobit-five/PyTorchTrainingKit',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'torch',  # need to verify torch version this works with
        'tqdm',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.6',  # need to verify this
)
