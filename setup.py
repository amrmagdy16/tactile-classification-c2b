from setuptools import find_packages, setup 
setup( name="dmrobotics", 
      version="0.1.4", 
      python_requires=">=3.8, <=3.11",
      packages=find_packages(), 
      license="", 
      author="Yipai Du", 
      author_email="yipai.du@outlook.com", 
      install_requires=[ 
          "numpy>=1.21,<=1.26", 
          "opencv-contrib-python>=4.6.0.66,<=4.11.0.84", 
          "scipy>=1.7.3,<=1.15.3", "setuptools>=45.2.0,<=80.1.0", 
          "cupy-cuda11x>=11.0.0", 
          'pyudev; platform_system=="Linux"', # For Linux platform 
          ], 
          package_data={"": ["*.so"]}, 
          include_package_data=True, 
          description="Tactile sensor interface for Daimon Robotics.", )