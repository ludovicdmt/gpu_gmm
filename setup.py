from setuptools import setup

setup(name='gpu_gmm',
      version='0.3',
      description='Online Gaussian Mixture and on GPU',
      url='Not yet on git',
      author='Ludovic DARMET',
      author_email='ludovic.darmet@gipsa-lab.fr',
      packages=['gpu_gmm'],
      install_requires=[
          'numpy', 'sklearn', 'tensorflow', 'pandas'
      ],
      zip_safe=False)