import setuptools

with open("README.md", 'r') as fh:
      long_description = fh.read()

setuptools.setup(
      name="easynlp",
      version="0.0.1",
      author="Ke Wang",
      author_email="wangke17@pku.edu.cn",
      description="An easy-to-use toolkit for natural language processing tasks.",
      long_description=long_description,
      url="https://github.com/Nrgeup/EasyNLP",
      packages=setuptools.find_packages(),
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
      ],
      zip_safe=False,
)
