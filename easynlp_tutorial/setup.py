import setuptools

with open("README.md", 'r') as fh:
      long_description = fh.read()

setuptools.setup(
      name="Easy-NLP",
      version="0.0.1",
      author="Ke Wang",
      author_email="wangke17@pku.edu.cn",
      description="An easy-to-use toolkit for natural language processing tasks.",
      long_description=long_description,
      url="https://github.com/Nrgeup/EasyNLP",
      
)





setup(name='easynlp',
      version='0.1',
      description='clinical trial information retriver',
      url='http://github.com/tongling/clinicaltrial',
      author='Ling',
      author_email='tonglingacademic@gmail.com',
      license='MIT',
      packages=['easynlp', 'filter'],
      zip_safe=False)
