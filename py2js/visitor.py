import ast
import functools
import typing as t


class VisitorContext:
    parent: ast.AST
    node: ast.AST


T = t.TypeVar('T', bound=VisitorContext)


class VisitorMethod:
    def __call__(self: 'Visitor', node: ast.AST, context: VisitorContext) -> str:
        pass


class Transformer(t.Generic[T]):
    def pre_transform(self, node: ast.AST, visitor: 'Visitor', context: T) -> t.List[ast.AST]:
        pass

    def post_transform(self, code: str, visitor: 'Visitor', context: T) -> t.Optional[str]:
        pass


class Visitor(t.Generic[T]):
    def __init__(self, context_factory_or_default: t.Union[T, t.Callable[[], T]], transformers: t.Optional[t.List[Transformer]] = None) -> None:
        self.create_context = context_factory_or_default if callable(context_factory_or_default) else lambda: context_factory_or_default
        self.transformers: t.List[Transformer] = transformers or []

    @functools.cache
    def get_visitor(self, node) -> t.Optional[VisitorMethod]:
        field_name: str = 'visit_' + node.__class__.__name__
        if hasattr(self, field_name):
            return getattr(self, field_name)
        elif hasattr(self, field_name := field_name.lower()):
            return getattr(self, field_name)

    def pre_transformers(self, node: ast.AST, context: T):
        for transformer in self.transformers:
            if res := transformer.pre_transform(node, self, context):
                node = res
        return node

    def post_transformers(self, code: str, context: T):
        for transformer in self.transformers:
            if res := transformer.post_transform(code, self, context):
                code = res
        return code

    def transform(self, node: ast.AST, context: t.Optional[T] = None):
        if context is None:
            context = self.create_context()
        else:
            context = context.copy(node=node)

        method = self.get_visitor(node)

        node = self.pre_transformers(node, context)

        if method is not None:
            visited = method(node, context)

            if not isinstance(visited, (list, tuple, set)):
                visited = self.post_transformers(visited, context)
            else:
                for i, code in enumerate(tuple(visited)):
                    visited[i] = self.post_transformers(code, context)

                visited = ';'.join(visited)

            return visited

        return f"/* {node.__class__.__name__} translator not found. */"

    def sperator(self, ast: t.Union[t.List[ast.AST], ast.AST], context: t.Optional[T] = None):
        return ';'

    def visit(self, ast: t.Union[t.List[ast.AST], ast.AST], context: t.Optional[T] = None) -> list[str]:
        if context is None:
            context = self.create_context()
        else:
            context = context.copy()

        if not isinstance(ast, (list, tuple, set, map, filter)):
            return self.transform(ast, context=context)
        else:
            return self.sperator(ast, context=context).join(map(lambda x: self.transform(x, context=context), ast))
