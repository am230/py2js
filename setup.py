from setuptools import setup, find_packages, find_namespace_packages

with open('README.md', mode='r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', mode='r', encoding='utf-8') as f:
    requires = map(str.strip, f.readlines())

author = 'am230'
name = 'py2js'
py_modules = [name]

setup(
    name=name,
    version="1.0.1",
    keywords=("javascript", "convert"),
    description="Write javascript in python with python syntax",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="MIT Licence",
    url=f"https://github.com/{author}/{name}",
    author=author,
    author_email="am.230@outlook.jp",
    py_modules=py_modules,
    platforms="any",
    packages=find_packages(),
    requires=requires
)