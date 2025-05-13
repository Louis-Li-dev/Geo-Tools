from setuptools import setup, find_packages

setup(
    name='geo_tools',
    version='0.1.10',
    description='A geospatial tool for looking up countries from coordinates.',
    author='An-Syu Li',
    author_email='yessir0621@gmail.com',
    packages=find_packages(include=['geo_tools', 'geo_tools.*']),  # Automatically discovers packages in your project
    install_requires=[
        'pandas==1.5.3',
        'numpy==1.26.4',
        'matplotlib',
        'seaborn',
        'folium',
        'geopandas',
        'basemap',
        'shapely',
        'requests',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: GIS',
    ],
    python_requires='>=3.6',
)
