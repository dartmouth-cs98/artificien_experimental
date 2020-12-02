from setuptools import find_packages, setup

setup(
        name='artificienlib',
        packages=find_packages(include=['artificienlib']),
        version='0.1.0',
        description='Library for artificien, abstracts federated learning processes',
        author='Jake Epstein',
        license='MIT',
        install_requires=[
            'syft==0.2.9',
            'websocket-client==0.57.0',
            'torch',
            'requests',
            'boto3'
        ],
        setup_requires=['pytest-runner'],
        tests_require=['pytest==4.4.1'],
        test_suite='tests',
)
