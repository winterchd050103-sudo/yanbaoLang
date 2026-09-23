"""安全计算工具：为什么不让 LLM 口算。

LLM 做「48.72 / 1.13 * 100」这类算术会出错（tokenizer 对数字不友好），
必须外置确定性计算器。这里用 AST 白名单求值而非 eval ——
这是安全边界：模型生成的字符串永远不直接进解释器。
"""

from __future__ import annotations

import ast
import operator as op

from langchain_core.tools import tool

# 仅放行四则与幂运算的二元/一元操作符
_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("指数过大")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不允许的表达式元素: {type(node).__name__}")


@tool
def safe_calc(expression: str) -> str:
    """计算算术表达式（仅支持 + - * / % ** 与括号）。

    涉及增长率、估值倍数、目标价空间等任何数字计算时必须使用本工具，
    不要心算。
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"{expression} = {result:g}"
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as e:
        return f"计算失败: {e}（只支持纯算术表达式，如 (128.5-102.3)/102.3*100）"
