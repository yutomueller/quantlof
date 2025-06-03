from setuptools import setup, find_packages

setup(
    name='quantlof',
    version='0.1.0',
    description='Quantum-enhanced Local Outlier Factor algorithm',
    author='Yuto Mueller',
    author_email='geoyuto@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'qiskit==2.0.2',
        'qiskit-aer==0.17.0',
        'qiskit-ibm-runtime==0.40.0',
        'scikit-learn==1.6.1',
        'scipy==1.15.3',
        'numpy==2.2.6',
        'dimod==0.12.20',
        'dwave-neal==0.6.0',
        'dwave-samplers==1.5.0',
    ],
    python_requires='>=3.8',
)
