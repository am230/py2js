import py2js

print('sample1\n', py2js.convert('''
def test():
    console.log('text')
'''))

def main():
    class Test:
        def log(text):
            console.log(text)

    class Main(Test):
        def __init__(self, a, b, *args) -> None:
            self.a = a
            self.b = b
            self.args = args
        
        def sum(self):
            self.log([self.a, self.b, *self.args].reduce(lambda sum, element: sum + element, 0))

    @console.log
    class Decorated:
        pass

    Main(1, 2, 3, 4, 5).sum()

print('sample2\n', py2js.convert(main))