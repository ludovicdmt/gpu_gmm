from setuptools import setup

setup(name='gpu_gmm',
      version='0.1',
      description='Online Gaussian Mixture and on GPU',
      url='Not yet on git',
      author='Ludovic DARMET',
      author_email='ludovic.darmet@gipsa-lab.fr',
      packages=['gpu_gmm'],
      install_requires=[
          'opencv-python', 'sklearn', 'tensorflow'
      ],
      zip_safe=False)