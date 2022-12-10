import ast
import dataclasses
import enum
import inspect
import types
import typing as t
from functools import reduce

import jsbeautifier
from strbuilder import BaseBuilder, SurroundBuilder

from .visitor import Transformer, Visitor, VisitorContext

transformers: t.List['Transformer'] = []

BoolOpTable = {
    ast.Or: '||',
    ast.And: '&&'
}
UnaryOpTable = {
    ast.Invert: '~',
    ast.Not: '!',
    ast.UAdd: '+',
    ast.USub: '-',
}
OperatorTable = {
    ast.Add: '+',
    ast.BitAnd: '&',
    ast.BitOr: '|',
    ast.BitXor: '^',
    ast.Div: '/',
    ast.FloorDiv: '//',
    ast.LShift: '<<',
    ast.Mod: '%',
    ast.Mult: '*',
    ast.Pow: '**',
    ast.RShift: '>>',
    ast.Sub: '-',
}
ConstantTable = {
    None: 'null',
    True: 'true',
    False: 'false'
}


class JSScope(enum.Enum):
    MODULE = 0
    CLASS_FIELD = 1
    FUNCTION_FIELD = 2
    ARGUMENT = 3


@dataclasses.dataclass
class JSVisitorContext(VisitorContext):
    # import_mode: ImportMode = dataclasses.field(default='import')
    locals_variables: t.Dict[str, type] = dataclasses.field(default_factory=dict)
    scope: JSScope = dataclasses.field(default=JSScope.MODULE)
    boolop_table: t.Dict[type, str] = dataclasses.field(default_factory=BoolOpTable.copy)
    unaryop_table: t.Dict[type, str] = dataclasses.field(default_factory=UnaryOpTable.copy)
    operator_table: t.Dict[type, str] = dataclasses.field(default_factory=OperatorTable.copy)
    constant_table: t.Dict[t.Any, str] = dataclasses.field(default_factory=ConstantTable.copy)
    parent: ast.AST = dataclasses.field(default=None)
    node: ast.AST = dataclasses.field(default=None)

    def copy(self, node: t.Optional[ast.AST] = None) -> 'JSVisitorContext':
        return JSVisitorContext(
            self.locals_variables,
            self.scope,
            self.boolop_table,
            self.unaryop_table,
            self.operator_table,
            self.constant_table,
            self.node,
            node or self.node
        )


class CodeGen(Visitor[JSVisitorContext]):
    def __init__(self) -> None:
        super().__init__(JSVisitorContext, transformers=transformers)

    def sperator(self, ast: t.Union[t.List[ast.AST], ast.AST], context: t.Optional[JSVisitorContext] = None):
        if context.scope == JSScope.ARGUMENT:
            return ','
        return ';'

    def visit_Module(self, node: ast.Module, ctx: JSVisitorContext):
        return self.visit(node.body, ctx)

    def visit_Import(self, node: ast.Import, ctx: JSVisitorContext):
        return [f'import {x.asname or x.name} from "{x.name}"' for x in node.names]

    def visit_ImportFrom(self, node: ast.ImportFrom, ctx: JSVisitorContext):
        multiple = len(node.names) > 1
        first = node.names[0]
        return BaseBuilder('import')\
            .write_if(multiple,
                      lambda: SurroundBuilder(surround='{}')
                      .write(BaseBuilder(separator=',').write(BaseBuilder(x.name).write_if(x.asname, BaseBuilder('as').write(x.asname)) for x in node.names)),
                      or_else=BaseBuilder(f'* as {node.module}') if first.name == '*' else SurroundBuilder(first.name, surround='{}').write_if(first.asname, BaseBuilder('as').write(first.asname))
                      )\
            .write('from')\
            .write(SurroundBuilder(node.module))\
            .build()

    def visit_ClassDef(self, node: ast.ClassDef, ctx: JSVisitorContext):
        """
        class MyPlugin extends Function {
            constructor(...args) {
                super()
                if ('__init__' in this) {this.__init__(this, ...args);}
                return new Proxy(this, {
                    apply(target, self, args) {
                        return target.__call__(self, ...args)
                    }
                })
            }

            aMethod() {
                console.log(this + ' b')
            }
        }
        function applyMixins(derivedCtor, baseCtors) {
            baseCtors.forEach(baseCtor => {
                Object.getOwnPropertyNames(baseCtor.prototype).forEach(name => {
                    if (name !== 'constructor') {
                        derivedCtor.prototype[name] = baseCtor.prototype[name];
                    }
                });
            });
        }
        """
        context = ctx.copy()
        context.scope = JSScope.CLASS_FIELD

        body = node.body

        builder = BaseBuilder()\
        .write(f'let {node.name} = class')\
        .write(SurroundBuilder(surround='{}')
                      .write('''constructor(...args) {if ('__init__' in this) this.__init__(this, ...args); return new Proxy(this, { apply: (target, self, args) => target.__call__(self, ...args), get: (target, prop, receiver) => {if (target[prop] instanceof Function) {return (...args) => target[prop](target, ...args)} else {return target[prop]}}})}''')
                      .write(self.visit(node.body, context))
                      .write(self.visit(filter(lambda node: not isinstance(node, ast.FunctionDef), body), context)))\
        .write_if(node.bases, (lambda: f'''Object.getOwnPropertyNames({base}.prototype).forEach(name => {{if (name !== 'constructor') {{{node.name}.prototype[name] = {base}.prototype[name];}}}});''' for base in map(lambda base: self.visit(base, ctx), node.bases)))\
        .write_if(node.decorator_list, (f'{node.name} = ', reduce(lambda value, element: f'{element}({value})', map(lambda decorator: self.visit(decorator, ctx), node.decorator_list), node.name), ';'))\
        .write(f"if (typeof {node.name} !== 'undefined') {{")\
        .write(f'{node.name} = new Proxy({node.name}')\
        .write(', { apply: (clazz, thisValue, args) => new clazz(...args) })')\
        .write('}')\

        return builder.build()

    def visit_FunctionDef(self, node: ast.FunctionDef, ctx: JSVisitorContext):
        context = ctx.copy()
        context.scope = JSScope.FUNCTION_FIELD

        if ctx.scope == JSScope.CLASS_FIELD:
            return BaseBuilder()\
                .write(node.name)\
                .write(f'{self.visit(node.args, ctx)}')\
                .write(SurroundBuilder(surround='{}')
                       .write(self.visit(node.body, context)))\
                .build()
        return BaseBuilder('let')\
            .write(node.name)\
            .write('=')\
            .write(f'{self.visit(node.args, ctx)} =>')\
            .write(SurroundBuilder(surround='{}')
                   .write(self.visit(node.body, context)))\
            .build()

    def visit_If(self, node: ast.If, ctx: JSVisitorContext):
        return BaseBuilder('if').write(SurroundBuilder(surround='()').write(self.visit(node.test, ctx))).write(SurroundBuilder(surround='{}').write(self.visit(node.body, ctx))).build()

    def visit_Compare(self, node: ast.Compare, ctx: JSVisitorContext):
        ops = BaseBuilder()
        invert = False
        for op in node.ops:
            t = type(op)
            if t == ast.Eq:
                ops.write('==')
            elif t == ast.Is:
                ops.write('===')
            elif t == ast.IsNot:
                ops.write('!==')
            elif t == ast.NotIn:
                ops.write('in')
                invert = True
            elif t == ast.In:
                ops.write('in')

        builder = BaseBuilder(self.visit(node.left, ctx)).write(ops).write(self.visit(node.comparators, ctx))

        if invert:
            return f'!({builder.build()})'

        return builder.build()

    def visit_Name(self, node: ast.Name, ctx: JSVisitorContext):
        return node.id

    def visit_Constant(self, node: ast.Constant, ctx: JSVisitorContext):
        if node.value in ctx.constant_table:
            return ctx.constant_table[node.value]
        if isinstance(node.value, str):
            return f'"{node.value}"'
        else:
            return f'{node.value}'

    def visit_Assign(self, node: ast.Assign, ctx: JSVisitorContext):
        return BaseBuilder().write_if(not any(map(lambda target: isinstance(target, ast.Attribute), node.targets)) and ctx.scope != JSScope.CLASS_FIELD, 'let').write(f'{self.visit(node.targets, ctx)} = {self.visit(node.value, ctx)}').build()

    def visit_Call(self, node: ast.Call, ctx: JSVisitorContext):
        context = ctx.copy()
        context.scope = JSScope.ARGUMENT

        starreds = tuple(filter(lambda arg: isinstance(arg, ast.Starred), node.args))
        args = filter(lambda arg: not isinstance(arg, ast.Starred), node.args)

        return BaseBuilder(f'{self.visit(node.func, ctx)}')\
            .write(SurroundBuilder(surround='()', separator=',')
                   .write_if(args, lambda: self.visit(args, context))
                   .write_if(node.keywords, lambda: SurroundBuilder(surround='{}', separator=',')
                                 .write(self.visit(filter(lambda kw: kw.arg is not None, node.keywords), context))
                                 .write(self.visit(filter(lambda kw: kw.arg is None, node.keywords), context)))
                   .write_if(starreds, lambda: self.visit(starreds, context))
                   ).build()

    def visit_keyword(self, node: ast.keyword, ctx: JSVisitorContext):
        return f'{node.arg}:{self.visit(node.value)}' if node.arg is not None else f'...{self.visit(node.value)}'

    def visit_arguments(self, node: ast.arguments, ctx: JSVisitorContext):
        context = ctx.copy()
        context.scope = JSScope.ARGUMENT

        kwlen = len(node.defaults)
        kwargs = node.args[-kwlen:]
        args = node.args[:len(node.args)-kwlen]
        keywords = dict(zip(kwargs, node.defaults))

        return SurroundBuilder(surround='()')\
            .write(
            BaseBuilder(separator=',')
            .write_if(node.args, self.visit(args, context))
            .write_if(node.kwarg or node.defaults, lambda:
                      SurroundBuilder(surround='{}', separator=',')
                      .write_if(keywords, lambda: (f'{self.visit(k, context)}={self.visit(v, context)}' for k, v in keywords.items()))
                      .write_if(node.kwarg, lambda: f'...{self.visit(node.kwarg, context)}')
                      .build())
            .write_if(node.vararg, lambda: f'...{self.visit(node.vararg, context)}')
            .build()
        ).build()

    def visit_Attribute(self, node: ast.Attribute, ctx: JSVisitorContext):
        return f'{self.visit(node.value, ctx)}.{node.attr}'

    def visit_Expr(self, node: ast.Expr, ctx: JSVisitorContext):
        return self.visit(node.value, ctx)

    def visit_NamedExpr(self, node: ast.NamedExpr, ctx: JSVisitorContext):
        return f'{self.visit(node.target)}={self.visit(node.value, ctx)}'

    def visit_arg(self, node: ast.arg, ctx: JSVisitorContext):
        return node.arg

    def visit_Return(self, node: ast.Return, ctx: JSVisitorContext):
        return f'return {self.visit(node.value, ctx)}'

    def visit_List(self, node: ast.List, ctx: JSVisitorContext):
        return SurroundBuilder(surround='[]', separator=',').write(map(lambda item: self.visit(item, ctx), node.elts)).build()

    def visit_Tuple(self, *args):
        return f'/*tuple*/{self.visit_List(*args)}'

    def visit_Lambda(self, node: ast.Lambda, ctx: JSVisitorContext):
        return BaseBuilder().write(f'{self.visit(node.args, ctx)}=>').write(self.visit(node.body, ctx)).build()

    def visit_For(self, node: ast.For, ctx: JSVisitorContext):
        return BaseBuilder(f'for (let _i=0,_iter={self.visit(node.iter)},_len=_iter.length; _i<_len; _i++)')\
            .write('{')\
            .write(BaseBuilder(separator=';')
                   .write(f'let {self.visit(node.target)} = _iter[i]')
                   .write(self.visit(node.body)))\
            .write('}').build()

    def visit_BoolOp(self, node: ast.BoolOp, ctx: JSVisitorContext):
        return ctx.boolop_table[type(node.op)].join(map(lambda value: self.visit(value, ctx), node.values))

    def visit_UnaryOp(self, node: ast.UnaryOp, ctx: JSVisitorContext):
        return f'{ctx.unaryop_table[type(node.op)]}{self.visit(node.operand, ctx)}'

    def visit_ListComp(self, node: ast.ListComp, ctx: JSVisitorContext):
        return f'{self.visit(node.generators, ctx)}'

    def visit_DictComp(self, node: ast.DictComp, ctx: JSVisitorContext):
        "Object.fromEntries([1,2,3].map(n=>[n,1]))"
        return f'Object.fromEntries({self.visit(node.generators, ctx)})'

    def visit_comprehension(self, node: ast.comprehension, ctx: JSVisitorContext):
        builder = BaseBuilder(f'{self.visit(node.iter, ctx)}')
        builder.write_if(node.ifs, (f'.filter({self.visit(x, ctx)})' for x in node.ifs))

        if isinstance(ctx.parent, ast.DictComp):
            builder.write(f'.map(({self.visit(node.target, ctx)})=>[{self.visit(ctx.parent.key, ctx)}, {self.visit(ctx.parent.value, ctx)}])')
        else:
            builder.write(f'.map(({self.visit(node.target, ctx)})=>{self.visit(ctx.parent.elt, ctx)})')

        return builder.build()

    def visit_IfExp(self, node: ast.IfExp, ctx: JSVisitorContext):
        return f'{self.visit(node.test, ctx)} ? {self.visit(node.body, ctx)} : {self.visit(node.orelse, ctx)}'

    def visit_Starred(self, node: ast.Starred, ctx: JSVisitorContext):
        return f'...{self.visit(node.value, ctx)}'

    def visit_BinOp(self, node: ast.BinOp, ctx: JSVisitorContext):
        return f'{self.visit(node.left, ctx)}{ctx.operator_table[type(node.op)]}{self.visit(node.right, ctx)}'

    def visit_AnnAssign(self, node: ast.AnnAssign, ctx: JSVisitorContext):
        return BaseBuilder().write_if(not isinstance(node.target, ast.Attribute) and ctx.scope != JSScope.CLASS_FIELD, 'let').write(f'{self.visit(node.target, ctx)}').write_if(node.value is not None, f'={self.visit(node.value, ctx)}').build()

    def visit_Pass(self, node: ast.Pass, ctx: JSVisitorContext):
        return '/* pass */'

    def visit_With(self, node: ast.With, ctx: JSVisitorContext):
        builder = SurroundBuilder(surround='{}', separator=';')
        builder.write(
            BaseBuilder('let')
            .write(SurroundBuilder(surround='[]', separator=',')
                   .write(f'_with_{i}' for i, expr in enumerate(node.items)))
            .write('=')
            .write(SurroundBuilder(surround='[]', separator=',').write(f'{self.visit(expr.context_expr)}' for expr in node.items))
        )\
            .write(BaseBuilder(separator=';').write(BaseBuilder().write_if(expr.optional_vars, f'{self.visit(expr.optional_vars)}=').write(f'_with_{i}.__enter__()') for i, expr in enumerate(node.items)))\
            .write(self.visit(node.body))\
            .write(BaseBuilder(separator=';').write(BaseBuilder().write_if(expr.optional_vars, f'{self.visit(expr.optional_vars)}=').write(f'_with_{i}.__exit__()') for i, expr in enumerate(node.items)))

        return builder.build()


def convert(
    source: t.Union[str, ast.AST, types.ModuleType, types.FunctionType],
    formatter: t.Optional[t.Callable[[str], str]] = jsbeautifier.beautify,
    code_gen: t.Optional[Visitor] = CodeGen
):
    generator = code_gen()

    if isinstance(source, ast.AST):
        parsed = source
    elif isinstance(source, str):
        parsed = ast.parse(source)
    else:
        parsed = ast.parse(inspect.getsource(source))

    return formatter(generator.visit(parsed))
