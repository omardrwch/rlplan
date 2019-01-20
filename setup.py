from setuptools import setup, find_packages

packages_ = find_packages()
packages = [p for p in packages_ if not(p == 'tests')]

setup(name='minigym',
      version='0.0.1-dev',
      description='',
      url='',
      author='Omar Darwiche Domingues',
      author_email='omar.drwch@gmail.com',
      license='MIT',
      packages=packages,
      install_requires=[],
      zip_safe=False)