import ast
import operator

from app.constants.tooling import (
    ALLOWED_CALC_CHARACTERS,
    CALC_EXPRESSION_PREVIEW_CHARS,
    MAX_CALC_ABS_VALUE,
    MAX_CALC_EXPRESSION_CHARS,
    WEATHER_BY_CITY,
)
from app.utils.logger import log_warning

ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def get_weather(city: str) -> str:
    return WEATHER_BY_CITY.get(city, f"{city}天气信息暂不可用。")


def _ensure_safe_number(value: int | float) -> int | float:
    if abs(value) > MAX_CALC_ABS_VALUE:
        raise ValueError("number too large")
    return value


def _eval_arithmetic_node(node: ast.AST) -> int | float:
    """只解释安全的基础算术 AST 节点，避免使用 eval。"""

    if isinstance(node, ast.Expression):
        return _eval_arithmetic_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _ensure_safe_number(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINARY_OPERATORS:
        left = _eval_arithmetic_node(node.left)
        right = _eval_arithmetic_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("division by zero")
        result = ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)
        return _ensure_safe_number(result)

    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY_OPERATORS:
        operand = _eval_arithmetic_node(node.operand)
        result = ALLOWED_UNARY_OPERATORS[type(node.op)](operand)
        return _ensure_safe_number(result)

    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    expression = (expression or "").strip()
    try:
        if not expression or len(expression) > MAX_CALC_EXPRESSION_CHARS:
            log_warning(
                "calculate",
                "unsupported expression length",
                {"expression_preview": expression[:CALC_EXPRESSION_PREVIEW_CHARS]},
            )
            return "不支持的表达式"
        if not all(c in ALLOWED_CALC_CHARACTERS for c in expression):
            log_warning(
                "calculate",
                "unsupported expression characters",
                {"expression_preview": expression[:CALC_EXPRESSION_PREVIEW_CHARS]},
            )
            return "不支持的表达式"
        parsed = ast.parse(expression, mode="eval")
        result = _eval_arithmetic_node(parsed)
        return str(result)
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        log_warning(
            "calculate",
            "arithmetic evaluation failed",
            {"error": f"{exc.__class__.__name__}: {exc}", "expression": expression},
        )
        return "计算失败"
