from setuptools import setup, find_packages

setup(
    name = 'AbX',
    version = '2.0.0',
    license='MIT',
    description = 'AbX - Guided Score based diffusion model for antibody design',    
    packages=find_packages(include=['abx', 'abx/*']),
    include_package_data=True,
    package_data={
        'abx': ['common/default_rigids.json', 'common/stereo_chemical_props.txt'],
    },
  
  install_requires=[],
)
