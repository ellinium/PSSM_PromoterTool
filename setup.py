from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='PSSM-promoter-tool',
      version='1.0.0',
      description='The tools applies direct and inverted Codon Restrained Promoter Silencing method to the provided gene sequence',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/ellinium/PSSM_PromoterTool',
      author='Ellina Trofimova',
      author_email='ellina.trofimova@gmail.com',
      license='GNU General Public Licensepython setup.py bdist_wheel',
      install_requires=['biopython', 'click', 'cloudpickle', 'dask', 'distributed', 'fsspec', 'importlib-metadata','joblib',
                        'locket', 'msgpack', 'numpy', 'packaging', 'pandas', 'partd', 'psutil', 'python-dateutil', 'pytz',
                        'scikit-learn', 'scipy', 'six', 'sortedcontainers', 'tblib', 'threadpoolctl', 'toolz', 'tornado', 'urllib3',
                        'zict', 'zipp'],
      py_modules=['PSSMPromoterCalculator'],
      zip_safe=False,
      keywords='promoter prediction transcription rate')


