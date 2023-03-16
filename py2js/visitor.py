import ast
import functools
import inspect
from typing import Callable, Generic, List, Optional, Self, TypeVar, Union


class VisitorContext:
    parent: ast.AST
    node: ast.AST


T = TypeVar('T', bound=VisitorContext)


class VisitorMethod:
    def __call__(self: Self, node: ast.AST, context: VisitorContext) -> str:
        pass


class Visitor(Generic[T]):
    def __init__(self, context_factory_or_default: Union[T, Callable[[], T]]) -> None:
        self.create_context = context_factory_or_default if callable(context_factory_or_default) else lambda: context_factory_or_default

    @functools.cache
    def get_visitor(self, node) -> Optional[VisitorMethod]:
        field_name: str = 'visit_' + node.__class__.__name__
        if hasattr(self, field_name):
            return getattr(self, field_name)
        elif hasattr(self, field_name := field_name.lower()):
            return getattr(self, field_name)

    def transform(self, node: ast.AST, context: Optional[T] = None):
        context = context or self.create_context()

        method = self.get_visitor(node)
        if method is None:
            return f"/* {node.__class__.__name__} translator not found. */"

        visited = method(node, context)
        if visited is None:
            raise ValueError(f'{method.__name__} returned null!\n{inspect.getsourcefile(method)}:{inspect.getsourcelines(method)[1]}')

        if isinstance(visited, (list, tuple, map, filter, set)):
            return ';'.join(visited)
        else:
            return visited

    def sperator(self, ast: Union[List[ast.AST], ast.AST], context: Optional[T] = None):
        return ';'

    def visit(self, ast: Union[List[ast.AST], ast.AST], context: Optional[T] = None) -> List[str]:
        if context is None:
            context = self.create_context()

        if not isinstance(ast, (list, tuple, set, map, filter)):
            return self.transform(ast, context=context)
        else:
            return self.sperator(ast, context=context).join(map(lambda x: self.transform(x, context=context), ast))
