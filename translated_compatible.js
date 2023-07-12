`class with self args`;
let Main = class {
    constructor(...args) {
        if ('__init__' in this) this.__init__(...args);
        return new Proxy(this, {
            apply: (target, self, args) => target.__call__(...args),
            get: (target, key) => target[key] || target.__getitem__(key)
        })
    };
    __init__(...__args) {
        ((self, value) => {
            self.a = value
        })(this, ...__args)
    };
    func(...__args) {
        ((self) => {
            console.log(self.a)
        })(this, ...__args)
    }
}
Main = new Proxy(Main, {
    apply: (clazz, thisValue, args) => new clazz(...args)
});;
Main(`hello, world!`).func()