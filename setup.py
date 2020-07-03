from setuptools import setup, find_packages

setup(
  name = 'slot_attention',
  packages = find_packages(),
  version = '0.0.6',
  license='MIT',
  description = 'Implementation of Slot Attention in Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/slot-attention',
  keywords = ['attention', 'artificial intelligence'],
  install_requires=[
      'torch'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)