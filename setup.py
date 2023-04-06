from setuptools import setup, find_packages

def parse_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

reqs = parse_requirements('requirements.txt')

VERSION = '2.0.10'
DESCRIPTION = 'Only Train Once (OTO): Automatic One-Shot General DNN Training and Compression Framework'
LONG_DESCRIPTION = 'Only Train Once (OTO): Automatic One-Shot General DNN Training and Compression (via Structured Pruning) Framework'

setup(
    name="only_train_once",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Tianyi Chen",
    author_email="tiachen@microsoft.com",
    license='MIT',
    packages=find_packages(),
    install_requires=reqs,
    url="https://github.com/tianyic/only_train_once",
    keywords='automatic, one-shot, structure pruning, sparse optimization',
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)