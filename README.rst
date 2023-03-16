Python to Javascript translator
===============================

.. image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :target: http://opensource.org/licenses/MIT
.. image:: https://badge.fury.io/py/py2js.svg
    :target: https://badge.fury.io/py/py2js

Usage
-----

.. code:: python

    import py2js

    @py2js.js
    def main():
        # code here


Options: py2js.js
-----------------

.. code:: python

    func:              t.Optional[t.Callable] 
    as_function:       t.Optional[bool]       
    python_compatible: t.Optional[bool]       

Examples
--------

python

.. code:: python

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

translated_normal.js
~~~~~~~~~~~~~~~~~~~~

.. code:: javascript

    `generator`;
    let generator_func = function*() {
        yield 1;
        yield 2;
        yield 3;
        yield*[9, 8, 7];
        return 0
    };
    for (let item of generator_func()) {
        console.log(item)
    };
    `for-else`;
    __else: {
        for (let i of [0, 1, 2, 3]) {
            if (i > 2) {
                break __else
            }
        }
        console.log(`else`)
    };
    `f-string`;
    let a = 4;
    console.log(`${a} fstring`);
    `class`;
    let Main = class {
        constructor() {
            this.a = 1
        };
        func() {
            console.log(this.a)
        }
        a
    }
    Main = new Proxy(Main, {
        apply: (clazz, thisValue, args) => new clazz(...args)
    });;
    Main().func();
    `try catch else finally`;
    __else: {
        try {
            throw SyntaxError(`syntax error`)
        } catch (__err) {
            if (__err instanceof SyntaxError) {
                e = __err;
                console.log(`syntax error raised`);
                break __else
            } {
                console.log(`excepted`);
                break __else
            }
        }
        console.log(`else`)
    };
    try {
        if (Boolean(1)) {
            throw SyntaxError(`syntax error`)
        }
    } catch (__err) {
        {
            /* pass */ }
    } finally {
        console.log(`finally`)
    };
    `while`;
    let i = 10;
    while (i > 0) {
        i -= 1
    };
    `comparator`;
    i = 5;
    if (0 < i < 9) {
        console.log(`true`)
    }

translated_compatible.js
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: javascript

    `class with self args`;
    let Main = class {
        constructor(...args) {
            if ('__init__' in this) this.__init__(...args);
            return new Proxy(this, {
                apply: (target, self, args) => target.__call__(...args),
                get: (target, key) => target.__getitem__(key)
            })
        }
        __init__ = (...__args) => {
            ((self, value) => {
                self.a = value
            })(this, ...__args)
        };
        func = (...__args) => {
            ((self) => {
                console.log(self.a)
            })(this, ...__args)
        }
    }
    Main = new Proxy(Main, {
        apply: (clazz, thisValue, args) => new clazz(...args)
    });;
    Main(`hello, world!`).func()

todo
----

match statement
