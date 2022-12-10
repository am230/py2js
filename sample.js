let main = () => {
    let Test = class {
        constructor(...args) {
            if ('__init__' in this) this.__init__(this, ...args);
            return new Proxy(this, {
                apply: (target, self, args) => target.__call__(self, ...args),
                get: (target, prop, receiver) => {
                    if (target[prop] instanceof Function) {
                        return (...args) => target[prop](target, ...args)
                    } else {
                        return target[prop]
                    }
                }
            })
        }
        log(text) {
            console.log(text)
        }
    }
    if (typeof Test !== 'undefined') {
        Test = new Proxy(Test, {
            apply: (clazz, thisValue, args) => new clazz(...args)
        })
    };
    let Main = class {
        constructor(...args) {
            if ('__init__' in this) this.__init__(this, ...args);
            return new Proxy(this, {
                apply: (target, self, args) => target.__call__(self, ...args),
                get: (target, prop, receiver) => {
                    if (target[prop] instanceof Function) {
                        return (...args) => target[prop](target, ...args)
                    } else {
                        return target[prop]
                    }
                }
            })
        }
        __init__(self, a, b, ...args) {
            self.a = a;
            self.b = b;
            self.args = args
        };
        sum(self) {
            self.log([self.a, self.b, ...self.args].reduce((sum, element) => sum + element, false))
        }
    }
    Object.getOwnPropertyNames(Test.prototype).forEach(name => {
        if (name !== 'constructor') {
            Main.prototype[name] = Test.prototype[name];
        }
    });
    if (typeof Main !== 'undefined') {
        Main = new Proxy(Main, {
            apply: (clazz, thisValue, args) => new clazz(...args)
        })
    };
    let Decorated = class {
        constructor(...args) {
            if ('__init__' in this) this.__init__(this, ...args);
            return new Proxy(this, {
                apply: (target, self, args) => target.__call__(self, ...args),
                get: (target, prop, receiver) => {
                    if (target[prop] instanceof Function) {
                        return (...args) => target[prop](target, ...args)
                    } else {
                        return target[prop]
                    }
                }
            })
        } /* pass */ /* pass */
    }
    Decorated = console.log(Decorated);
    if (typeof Decorated !== 'undefined') {
        Decorated = new Proxy(Decorated, {
            apply: (clazz, thisValue, args) => new clazz(...args)
        })
    };
    Main(true, 2, 3, 4, 5).sum()
}