import ast, operator
from pathlib import Path

_ALLOWED_BINOPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow}
_ALLOWED_UNARY={ast.UAdd:operator.pos,ast.USub:operator.neg}

def _eval_node(node):
    if isinstance(node, ast.Expression): return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value,(int,float)): return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS: return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left),_eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY: return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError("Only basic arithmetic is allowed.")

def calculator(expression:str)->str:
    if len(expression)>200: raise ValueError("Expression too long.")
    result=_eval_node(ast.parse(expression,mode="eval"))
    if isinstance(result,float): result=round(result,10)
    return str(result)

def list_documents(upload_dir:str)->list[str]:
    path=Path(upload_dir); path.mkdir(parents=True,exist_ok=True)
    return sorted(p.name for p in path.iterdir() if p.is_file())

TOOL_DEFINITIONS=[
 {"type":"function","function":{"name":"calculator","description":"Safely evaluate a basic arithmetic expression.","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"],"additionalProperties":False}}},
 {"type":"function","function":{"name":"list_documents","description":"List documents uploaded to the RAG assistant.","parameters":{"type":"object","properties":{},"additionalProperties":False}}}
]
