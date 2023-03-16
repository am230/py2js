import py2js
import typing

this = typing.Self


@py2js.js
def translated_normal():
    'generator'
    def generator_func():
        yield 1
        yield 2
        yield 3
        yield from [9, 8, 7]
        return 0

    for item in generator_func():
        console.log(item)

    'for-else'
    for i in [0, 1, 2, 3]:
        if i > 2:
            break
    else:
        console.log('else')

    'f-string'
    a = 4
    console.log(f'{a} fstring')

    'class'
    class Main:
        a: int

        def constructor():
            this.a = 1

        def func():
            console.log(this.a)

    Main().func()

    'try catch else finally'
    try:
        raise SyntaxError('syntax error')
    except SyntaxError as e:
        console.log('syntax error raised')
    except:
        console.log('excepted')
    else:
        console.log('else')

    try:
        if Boolean(1):
            raise SyntaxError('syntax error')
    except:
        pass
    finally:
        console.log('finally')

    'while'
    i = 10
    while i > 0:
        i -= 1

    'comparator'
    i = 5
    if 0 < i < 9:
        console.log('true')


@py2js.js(python_compatible=True)
def translated_compatible():
    'class with self args'
    class Main:
        def __init__(self, value):
            self.a = value

        def func(self):
            console.log(self.a)

    Main('hello, world!').func()


with open('translated_normal.js', 'w') as f:
    f.write(translated_normal)

with open('translated_compatible.js', 'w') as f:
    f.write(translated_compatible)
