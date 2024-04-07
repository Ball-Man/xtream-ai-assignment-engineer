from setuptools import setup

README = open('README.md').read()
REQUIREMENTS_BASE = open('requirements.txt').read().splitlines()

# Optional dependencies
REQUIREMENTS_NB = open(
    'requirements_nb.txt').read().splitlines()
REQUIREMENTS_WEB = open(
    'requirements_web.txt').read().splitlines()
REQUIREMENTS_ALL = (REQUIREMENTS_BASE + REQUIREMENTS_WEB + REQUIREMENTS_NB)

setup(name='diamond',
      classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      python_requires='>=3.9',
      version='1.0.0',
      description='Diamond price prediction tools.',
      long_description_content_type='text/markdown',
      long_description=README,
      url='https://github.com/Ball-Man/xtream-ai-assignment-engineer',
      # author='Francesco Mistri',
      # author_email='franc.mistri@gmail.com',
      license='MIT',
      packages=['diamond'],
      extras_require={
            'nb': REQUIREMENTS_NB,
            'web': REQUIREMENTS_WEB,
            'all': REQUIREMENTS_ALL},
      install_requires=REQUIREMENTS_BASE
      )
