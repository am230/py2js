from setuptools import setup, find_packages

with open('README.md', mode='r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', mode='r', encoding='utf-8') as f:
    requires = f.readlines()

setup(
    name="py2js",
    version="1.0.0",
    keywords=("javascript", "py2js"),
    description="a fast and simple micro-framework for small web applications",
    long_description=long_description,
    long_description_content_type='text/markdown',  # This is important!
    license="MIT Licence",
    url="https://github.com/am230/py2js",
    author="am230",
    author_email="am.230@outlook.jp",
    py_modules=['py2js'],
    platforms="any",
    requires=requires,
    packages=find_packages()
)