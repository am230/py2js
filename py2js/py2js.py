import ast
import enum
import inspect
import re
import types
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, List, Dict, Optional, Union, Callable

import jsbeautifier
from strbuilder import Builder, SurroundBuilder

from .visitor import Visitor, VisitorContext


class JSScope(enum.Enum):
    MODULE = 0
    CLASS_FIELD = 1
    FUNCTION_FIELD = 2
    ARGUMENT = 3
    F_STRING = 4
    FOR_LOOP = 5


@dataclass
class JSVisitorContext(VisitorContext):
    variables: List[str] = field(default_factory=list)
    global_variables: list[str] = field(default_factory=list)
    scope: JSScope = field(default=JSScope.MODULE)
    break_suffix: str = field(default='')
    parent: ast.AST = field(default=None)
    node: ast.AST = field(default=None)
    has_yield: bool = field(default=False)

    def is_existed(self, name: str) -> bool:
        return name in self.variables or name in self.global_variables

    def copy(
        self,
        variables: Dict[str, Any] = None,
        node: Optional[ast.AST] = None,
        scope: Optional[JSScope] = None,
        break_suffix: Optional[str] = None,
        has_yield: Optional[bool] = None,
        root: Optional['JSVisitorContext'] = None
    ) -> 'JSVisitorContext':
        return JSVisitorContext(
            variables=variables or self.variables,
            global_variables=self.global_variables,
            scope=scope or self.scope,
            break_suffix=self.break_suffix if break_suffix is None else break_suffix,
            parent=self.node,
            node=node or self.node,
            has_yield=self.has_yield if has_yield is None else has_yield
        )


class ConstantConsumer:
    def __call__(self, value: Any, ctx: JSVisitorContext) -> str: ...


class CodeGen(Visitor[JSVisitorContext]):
    BOOL_OP_TABLE: Dict[type, str] = {
        ast.Or: '||',
        ast.And: '&&'
    }

    UNARY_OP_TABLE: Dict[type, str] = {
        ast.Invert: '~',
        ast.Not: '!',
        ast.UAdd: '+',
        ast.USub: '-',
    }

    COMPARE_OP_TABLE: Dict[type, str] = {
        ast.Eq: '==',
        ast.NotEq: '!==',
        ast.Is: '===',
        ast.Gt: '>',
        ast.GtE: '>=',
        ast.Lt: '<',
        ast.LtE: '<=',
        ast.IsNot: '!==',
        ast.NotIn: 'in',
        ast.In: 'in'
    }

    OPERATOR_OP_TABLE: Dict[type, str] = {
        ast.Add: '+',
        ast.BitAnd: '&',
        ast.BitOr: '|',
        ast.BitXor: '^',
        ast.Div: '/',
        ast.FloorDiv: '/',
        ast.LShift: '<<',
        ast.Mod: '%',
        ast.Mult: '*',
        ast.Pow: '**',
        ast.RShift: '>>',
        ast.Sub: '-',
    }

    CONSTANT_CONSUMER_TABLE: Dict[type, ConstantConsumer] = {
        bool: lambda value, ctx: 'true' if value else 'false',
        str: lambda value, ctx: ('%s' if ctx.scope == JSScope.F_STRING else f'`%s`') % value.replace('`', '\\`').replace('$', '\\$'),
        int: lambda value, ctx: str(value),
        float: lambda value, ctx: str(value),
        type(None): lambda _, ctx: 'null',
        type(...): lambda _, ctx: 'ellipsis'
    }

    def __init__(self, python_compatible: bool = False) -> None:
        self.python_compatible = python_compatible
        super().__init__(JSVisitorContext)

    def sperator(self, ast: Union[List[ast.AST], ast.AST], context: Optional[JSVisitorContext] = None):
        if context.scope == JSScope.ARGUMENT:
            return ','
        return ';'

    def visit_Module(self, node: ast.Module, ctx: JSVisitorContext):
        return self.visit(node.body, ctx)

    def visit_Import(self, node: ast.Import, ctx: JSVisitorContext):
        return [f'import * as {x.asname} from "{x.name}"' if x.asname else f'import "{x.name}"' for x in node.names]

    def visit_ImportFrom(self, node: ast.ImportFrom, ctx: JSVisitorContext):
        builder = Builder(separator=',')
        for item in node.names:
            if item.name == '*':
                builder.write(f'/* Wildcard import not supported: {node.module} */')
                continue
            builder.write(f'{item.name} as {item.asname}' if item.asname else item.name)
        dir = '"../"'
        return f'import {{{builder.build()}}} from {node.module or dir}'

    def visit_ClassDef(self, node: ast.ClassDef, ctx: JSVisitorContext):
        context = ctx.copy(scope=JSScope.CLASS_FIELD)

        body = node.body
        name = ast.Name(node.name)
        builder = Builder()\
            .write(f'let {node.name} = class')\
            .write(SurroundBuilder(surround='{}')
                   .write_if(self.python_compatible, lambda: '''constructor(...args) {if ('__init__' in this) this.__init__(...args); return new Proxy(this, { apply: (target, self, args) => target.__call__(...args), get: (target, key) => target.__getitem__(key) })}''')
                   .write(self.visit(filter(lambda node: isinstance(node, ast.FunctionDef), body), context))
                   .write(self.visit(filter(lambda node: not isinstance(node, ast.FunctionDef), body), context))
                   )\
            .write_if(node.bases, lambda: (f'''Object.getOwnPropertyNames({base}.prototype).forEach(name => {{if (name !== 'constructor') {{{node.name}.prototype[name] = {base}.prototype[name];}}}});''' for base in map(lambda base: self.visit(base, ctx), node.bases)))\
            .write(f'{node.name} = new Proxy({node.name}, {{ apply: (clazz, thisValue, args) => new clazz(...args) }});')\
            .write_if(node.decorator_list, lambda: (f'{node.name} = ', reduce(lambda value, element: f'{element}({value})', map(lambda decorator: self.visit(decorator, ctx), node.decorator_list), node.name), ';'))

        return builder.build()

    def visit_FunctionDef(self, node: ast.FunctionDef, ctx: JSVisitorContext):
        context = ctx.copy(scope=JSScope.FUNCTION_FIELD)

        body = self.visit(node.body, context)
        if ctx.scope == JSScope.CLASS_FIELD:
            builder = Builder()
            builder.write_if(context.has_yield, '*')
            builder.write(node.name)
            if self.python_compatible:
                (
                    builder
                    .write('=(...__args)=>{')
                    .write(f'({self.visit(node.args, ctx)} =>')
                    .write(SurroundBuilder(surround='{}')
                           .write(body))
                    .write(')(this,...__args)}')
                )
            else:
                (
                    builder
                    .write(self.visit(node.args, ctx))
                    .write(SurroundBuilder(surround='{}')
                           .write(body))
                )
            return builder.build()
        builder = Builder()\
            .write_if(not ctx.is_existed(node.name), 'let')\
            .write(node.name)\
            .write('=')\
            .write_if(context.has_yield, 'function*')\
            .write(f'{self.visit(node.args, ctx)}')\
            .write_if(not context.has_yield, '=>')\
            .write(SurroundBuilder(surround='{}')
                   .write(body))
        ctx.variables.append(node.name)
        return builder.build()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef, ctx: JSVisitorContext):
        context = ctx.copy(scope=JSScope.FUNCTION_FIELD)

        body = self.visit(node.body, context)

        if ctx.scope == JSScope.CLASS_FIELD:
            return Builder()\
                .write_if(context.has_yield, '*')\
                .write(node.name)\
                .write(f'{self.visit(node.args, ctx)}')\
                .write(SurroundBuilder(surround='{}')
                       .write(body))\
                .build()
        builder = Builder()\
            .write_if(not ctx.is_existed(node.name), 'let')\
            .write(node.name)\
            .write('=')\
            .write_if(context.has_yield, 'async function*', or_else='async')\
            .write(f'{self.visit(node.args, ctx)}')\
            .write_if(not context.has_yield, '=>')\
            .write(SurroundBuilder(surround='{}')
                   .write(body))
        ctx.variables.append(node.name)
        return builder.build()

    def visit_Yield(self, node: ast.Yield, ctx: JSVisitorContext):
        ctx.has_yield = True
        return f'yield {self.visit(node.value, ctx)}'

    def visit_YieldFrom(self, node: ast.YieldFrom, ctx: JSVisitorContext):
        ctx.has_yield = True
        return f'yield* {self.visit(node.value, ctx)}'

    def visit_If(self, node: ast.If, ctx: JSVisitorContext):
        return Builder('if').write(SurroundBuilder(surround='()').write(self.visit(node.test, ctx))).write(SurroundBuilder(surround='{}').write(self.visit(node.body, ctx))).build()

    def visit_Compare(self, node: ast.Compare, ctx: JSVisitorContext):
        ops = Builder()
        invert = False
        for op in node.ops:
            op_type = type(op)
            ops.write(self.COMPARE_OP_TABLE[op_type])
            if op_type == ast.NotIn:
                invert = True

        builder = Builder(self.visit(node.left, ctx))
        comparators = node.comparators
        while comparators:
            builder.write(ops.pop(0))
            builder.write(self.visit(comparators.pop(0)))
        if not invert:
            return builder.build()
        return f'!({builder.build()})'

    def visit_Name(self, node: ast.Name, ctx: JSVisitorContext):
        return node.id

    def visit_Constant(self, node: ast.Constant, ctx: JSVisitorContext):
        return self.CONSTANT_CONSUMER_TABLE[type(node.value)](node.value, ctx)

    def visit_Assign(self, node: ast.Assign, ctx: JSVisitorContext):
        builder = Builder(separator=';')

        defines = [target.id for target in node.targets if isinstance(target, ast.Name) and not ctx.is_existed(target.id)]
        ctx.variables.extend(defines)

        if ctx.scope != JSScope.CLASS_FIELD:
            if len(node.targets) == len(defines):
                builder.write(f'let {", ".join(defines)} = {self.visit(node.value, ctx)}')
                return builder.build()
            elif defines:
                builder.write(f'let {", ".join(defines)}')
        builder.write(f'{self.visit(node.targets, ctx)} = {self.visit(node.value, ctx)}')
        return builder.build()
        # return Builder().write_if(node.targets in ctx.variables and ctx.scope != JSScope.CLASS_FIELD and not any(map(lambda target: isinstance(target, ast.Attribute), node.targets)), 'let').write(f'{self.visit(node.targets, ctx)} = {self.visit(node.value, ctx)}').build()

    def visit_Call(self, node: ast.Call, ctx: JSVisitorContext):
        context = ctx.copy(scope=JSScope.ARGUMENT)

        starreds = tuple(filter(lambda arg: isinstance(arg, ast.Starred), node.args))
        args = filter(lambda arg: not isinstance(arg, ast.Starred), node.args)

        return Builder(f'{self.visit(node.func, ctx)}')\
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
        context = ctx.copy(scope=JSScope.ARGUMENT)

        kwlen = len(node.defaults)
        kwargs = node.args[-kwlen:]
        args = node.args[:len(node.args)-kwlen]
        keywords = dict(zip(kwargs, node.defaults))

        return SurroundBuilder(surround='()')\
            .write(
            Builder(separator=',')
            .write_if(node.args, self.visit(args, context))
            .write_if(node.kwarg or node.defaults, lambda:
                      SurroundBuilder(surround='{}', separator=',')
                      .write_if(keywords, lambda: (f'{self.visit(key, context)}={self.visit(value, context)}' for key,  value in keywords.items()))
                      .write_if(node.kwarg, lambda: f'...{self.visit(node.kwarg, context)}')
                      .build() + '={}')
            .write_if(node.vararg, lambda: f'...{self.visit(node.vararg, context)}')
            .build()
        ).build()

    def visit_Attribute(self, node: ast.Attribute, ctx: JSVisitorContext):
        return f'{self.visit(node.value, ctx)}.{node.attr}'

    def visit_Expr(self, node: ast.Expr, ctx: JSVisitorContext):
        return self.visit(node.value, ctx)

    def visit_NamedExpr(self, node: ast.NamedExpr, ctx: JSVisitorContext):
        return f'{self.visit(node.target, ctx)}={self.visit(node.value, ctx)}'

    def visit_arg(self, node: ast.arg, ctx: JSVisitorContext):
        return node.arg

    def visit_Return(self, node: ast.Return, ctx: JSVisitorContext):
        return f'return {self.visit(node.value, ctx)}'

    def visit_List(self, node: ast.List, ctx: JSVisitorContext):
        return SurroundBuilder(surround='[]', separator=',').write(map(lambda item: self.visit(item, ctx), node.elts)).build()

    def visit_Tuple(self, *args):
        return f'/*tuple*/{self.visit_List(*args)}'

    def visit_Dict(self, node: ast.Dict, ctx: JSVisitorContext):
        return '{%s}' % ', '.join(f'{self.visit(key, ctx.copy(scope=JSScope.F_STRING))}: {self.visit(value, ctx)}' for key, value in zip(node.keys, node.values))

    def visit_Lambda(self, node: ast.Lambda, ctx: JSVisitorContext):
        return f'{self.visit(node.args, ctx)}=> {{return {self.visit(node.body, ctx)}}}'

    def visit_For(self, node: ast.For, ctx: JSVisitorContext):
        target = Builder()
        if isinstance(node.target, ast.Name):
            if not ctx.is_existed(node.target.id):
                target.write(f'let')
        target.write(self.visit(node.target, ctx))

        return (
            Builder()
            .write_if(node.orelse, '__else: {')
            .write(f'for ({target.build()} of {self.visit(node.iter, ctx)})')
            .write(f"{{{self.visit(node.body, ctx.copy(break_suffix='__else' if node.orelse else ''))}}}")
            .write_if(node.orelse, f'{self.visit(node.orelse, ctx)} }}')
        ).build()

    def visit_AsyncFor(self, node: ast.AsyncFor, ctx: JSVisitorContext):
        return (
            Builder()
            .write_if(node.orelse, '__else: {')
            .write(f'for await ({self.visit(node.target, ctx)} of {self.visit(node.iter, ctx)})')
            .write(f"{{{self.visit(node.body, ctx.copy(break_suffix='__else' if node.orelse else ''))}}}")
            .write_if(node.orelse, f'{self.visit(node.orelse, ctx)} }}')
        ).build()

    def visit_Await(self, node: ast.Await, ctx: JSVisitorContext):
        return f'await {self.visit(node.value, ctx)}'

    def visit_BoolOp(self, node: ast.BoolOp, ctx: JSVisitorContext):
        return self.BOOL_OP_TABLE[type(node.op)].join(map(lambda value: self.visit(value, ctx), node.values))

    def visit_UnaryOp(self, node: ast.UnaryOp, ctx: JSVisitorContext):
        return f'{self.UNARY_OP_TABLE[type(node.op)]}{self.visit(node.operand, ctx)}'

    def visit_ListComp(self, node: ast.ListComp, ctx: JSVisitorContext):
        ctx.parent = node
        return f'{self.visit(node.generators, ctx)}'

    def visit_DictComp(self, node: ast.DictComp, ctx: JSVisitorContext):
        "Object.fromEntries([1,2,3].map(n=>[n,1]))"
        return f'Object.fromEntries({self.visit(node.generators, ctx)})'

    def visit_comprehension(self, node: ast.comprehension, ctx: JSVisitorContext):
        builder = Builder(f'{self.visit(node.iter, ctx)}')
        target = self.visit(node.target, ctx)
        builder.write_if(node.ifs, (f'.filter({target}=>{self.visit(x, ctx)})' for x in node.ifs))

        if isinstance(ctx.parent, ast.DictComp):
            builder.write(f'.map(({target})=>[{self.visit(ctx.parent.key, ctx)}, {self.visit(ctx.parent.value, ctx)}])')
        else:
            builder.write(f'.map(({target})=>{self.visit(ctx.parent.elt, ctx)})')

        return builder.build()

    def visit_IfExp(self, node: ast.IfExp, ctx: JSVisitorContext):
        return f'{self.visit(node.test, ctx)} ? {self.visit(node.body, ctx)} : {self.visit(node.orelse, ctx)}'

    def visit_Starred(self, node: ast.Starred, ctx: JSVisitorContext):
        return f'...{self.visit(node.value, ctx)}'

    def visit_BinOp(self, node: ast.BinOp, ctx: JSVisitorContext):
        op_type = type(node.op)
        expr = f'{self.visit(node.left, ctx)}{self.OPERATOR_OP_TABLE[op_type]}{self.visit(node.right, ctx)}'
        if op_type == ast.FloorDiv:
            return f'(({expr}) >> 0)'
        return f'({expr})'

    def visit_AnnAssign(self, node: ast.AnnAssign, ctx: JSVisitorContext):
        if isinstance(node.target, ast.Name):
            ctx.variables.append(node.target.id)
        return (
            Builder()
            .write_if(not isinstance(node.target, ast.Attribute) and ctx.scope != JSScope.CLASS_FIELD, 'let')
            .write(f'{self.visit(node.target, ctx)}')
            .write_if(node.value is not None, lambda: f'={self.visit(node.value, ctx)}')
        ).build()

    def visit_Pass(self, node: ast.Pass, ctx: JSVisitorContext):
        return '/* pass */'

    def visit_With(self, node: ast.With, ctx: JSVisitorContext):
        builder = SurroundBuilder(surround='{}', separator=';')
        builder.write(
            Builder('let')
            .write(SurroundBuilder(surround='[]', separator=',')
                   .write(f'__with_{i}' for i, expr in enumerate(node.items)))
            .write('=')
            .write(SurroundBuilder(surround='[]', separator=',').write(f'{self.visit(expr.context_expr)}' for expr in node.items))
        )\
            .write(Builder(separator=';').write(Builder().write_if(expr.optional_vars, f'{self.visit(expr.optional_vars)}=').write(f'__with_{i}.__enter__()') for i, expr in enumerate(node.items)))\
            .write(self.visit(node.body))\
            .write(Builder(separator=';').write(Builder().write_if(expr.optional_vars, f'{self.visit(expr.optional_vars)}=').write(f'__with_{i}.__exit__()') for i, expr in enumerate(node.items)))

        return builder.build()

    def visit_JoinedStr(self, node: ast.JoinedStr, ctx: JSVisitorContext):
        context = ctx.copy(scope=JSScope.F_STRING)
        return '`{}`'.format(''.join(map(lambda item: self.visit(item, context), node.values)))

    def visit_FormattedValue(self, node: ast.FormattedValue, ctx: JSVisitorContext):
        return f'${{{self.visit(node.value, ctx)}}}'

    def visit_Delete(self, node: ast.Delete, ctx: JSVisitorContext):
        return ';'.join(map(lambda target: f'delete {self.visit(target, ctx)}', node.targets))

    def visit_AugAssign(self, node: ast.AugAssign, ctx: JSVisitorContext):
        op_type = type(node.op)
        target = self.visit(node.target, ctx)
        expr = f'{target} {self.OPERATOR_OP_TABLE[type(node.op)]}= {self.visit(node.value, ctx)}'
        if op_type == ast.FloorDiv:
            return f'{expr};{target} = ({target}) >> 0'
        return expr

    def visit_Subscript(self, node: ast.Subscript, ctx: JSVisitorContext):
        slice = node.slice
        if isinstance(slice, ast.Slice):
            if slice.step is None:
                return f'{self.visit(node.value, ctx)}.slice({self.visit(slice.lower, ctx)}, {self.visit(slice.upper, ctx)})'

        return f'{self.visit(node.value, ctx)}[{self.visit(slice, ctx)}]'

    def visit_Slice(self, node: ast.Slice, ctx: JSVisitorContext):
        return f'slice({self.visit(node.lower, ctx)}, {self.visit(node.upper, ctx)}, {self.visit(node.step, ctx)})'

    def visit_While(self, node: ast.While, ctx: JSVisitorContext):
        return 'while (%s) {%s}' % (self.visit(node.test, ctx), self.visit(node.body, ctx))

    def visit_Raise(self, node: ast.Raise, ctx: JSVisitorContext):
        return f'throw {self.visit(node.exc, ctx)}'

    def visit_Break(self, node: ast.Break, ctx: JSVisitorContext):
        return f'break {ctx.break_suffix}'

    def visit_Continue(self, node: ast.Continue, ctx: JSVisitorContext):
        return 'continue'

    def visit_Assert(self, node: ast.Assert, ctx: JSVisitorContext):
        return f'console.assert({self.visit(node.test, ctx)})'

    def visit_Try(self, node: ast.Try, ctx: JSVisitorContext):
        return (
            Builder(separator='')
            .write_if(node.orelse, '__else: {')
            .write(f'try {{{self.visit(node.body, ctx)}}}')
            .write_if(node.handlers, "catch(__err) {%s}" % '\n'.join(
                Builder()
                .write_if(handler.type, lambda: f'if (__err instanceof {self.visit(handler.type, ctx)})')
                .write(SurroundBuilder(surround='{}', separator=';')
                       .write_if(handler.name, f'{handler.name} = __err')
                       .write(self.visit(handler.body, ctx))
                       .write_if(node.orelse, 'break __else')
                       )
                .build()
                for handler in node.handlers
            ))
            .write_if(node.finalbody, Builder(f'finally {{{self.visit(node.finalbody, ctx)}').write('}'))
            .write_if(node.orelse, f'{self.visit(node.orelse, ctx)}}}')
        ).build()

    def visit_Match(self, node: ast.Match, ctx: JSVisitorContext):
        return '/*@deprecated match not supported */'

    def visit_Global(self, node: ast.Global, ctx: JSVisitorContext):
        ctx.global_variables.extend(node.names)
        return f'var {",".join(node.names)}'
    
    def visit_NoneType(self, null: None, ctx: JSVisitorContext):
        return 'null'


space_regex = re.compile('\s+')


def js(func: Optional[Callable] = None, as_function: bool = False, python_compatible: bool = False) -> str:
    def wrapper(func):
        lines = inspect.getsourcelines(func)[0]

        if not as_function:
            while lines:
                line = lines[0].strip()
                if line.startswith('@'):
                    lines.pop(0)
                elif line.startswith('def'):
                    lines.pop(0)
                    break
                else:
                    break

        if not lines:
            return ''

        match = space_regex.match(lines[0]).group() if lines[0].startswith(' ') else ''
        if match:
            spaces = len(match)
            lines = list(map(lambda line: line[spaces:], lines))

        return convert(''.join(lines), python_compatible=python_compatible)

    if func is None:
        return wrapper

    return wrapper(func)


def convert(
    source: Union[str, ast.AST, types.ModuleType, types.FunctionType],
    formatter: Optional[Callable[[str], str]] = jsbeautifier.beautify,
    code_gen: Optional[Visitor] = None,
    python_compatible: bool = False
):
    generator = code_gen or CodeGen(python_compatible=python_compatible)

    if isinstance(source, ast.AST):
        parsed = source
    elif isinstance(source, str):
        parsed = ast.parse(source)
    else:
        parsed = ast.parse(inspect.getsource(source))

    return formatter(generator.visit(parsed))
