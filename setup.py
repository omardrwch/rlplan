from setuptools import setup, find_packages

packages_ = find_packages()
packages = [p for p in packages_ if not(p == 'tests')]

setup(name='rlplan',
      version='0.0.1-dev',
      description='RL & Planning in MDPs',
      url='https://github.com/omardrwch/rlplan',
      author='Omar D. Domingues',
      author_email='',
      license='MIT',
      packages=packages,
      install_requires=['numpy', 'gym', 'pytest', 'matplotlib'],
      zip_safe=False)