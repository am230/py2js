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