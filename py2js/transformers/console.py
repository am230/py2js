# from ..js2py import transformers, Transformer, Visitor, JSVisitorContext, ast, t

# class PrintTransformer(Transformer):
#     def pre_transform(self, node: ast.AST, visitor: Visitor, context: JSVisitorContext) -> t.List[ast.AST]:
#         if not isinstance(node, ast.Call): return
#         if not isinstance(node.func, ast.Name): return
        
#         if node.func.id == 'print':
#             'Call(expr func, expr* args, keyword* keywords)'
#             'Name(identifier id, expr_context ctx)'
#             return ast.Call(ast.Name('console.log'), node.args, node.keywords)


# transformers.append(PrintTransformer())