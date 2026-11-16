"""Small offline environments. No network, Python eval, shell, or model API."""

import ast
import math
import operator
import re

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def calculate(expression: str) -> str:
    if len(expression) > 100:
        raise ValueError("Expression too long")
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 30:
        raise ValueError("Expression too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = float(node.value)
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            value = OPS[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        else:
            raise ValueError("Only numbers and + - * / are supported")
        if not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("Result out of bounds")
        return value

    return format(visit(tree.body), ".12g")


def tool_response(text: str, task: str) -> str | None:
    tag = {"calculator": "calc", "retrieval": "search"}.get(task)
    if tag is None:
        return None
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if not match:
        return None
    try:
        if task == "calculator":
            return calculate(match[1])
        station = re.fullmatch(r"Station-(\d{3})", match[1].strip())
        if station and int(station[1]) < 100:
            return f"{match[1].strip()}: access code {1000 + int(station[1]) * 17}"
        return "No matching local record"
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
        return "Tool error: invalid expression"
