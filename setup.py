from distutils.core import setup

setup(
    name='mixedlogistic',
    version='0.1.0',
    author='Pan Chao',
    author_email='panc@purdue.edu',
    url='http://stat.purdue.edu/~panc',
    packages=['data',
              'examples',
              'mixedlogistic',
              'mlp'],
    long_description=open("README.md").read(),
    install_requires=[
        'numpy',
        'scipy',
        'theano'
    ]
)
