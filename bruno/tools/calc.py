"""A calculator for arithmetic the user states outright.

Built on the hypothesis that the 3B's 37.5% confident-wrong rate on the hard tutor
set came from bad arithmetic, and that it recalled formulas better than it
evaluated them. **Measured, that hypothesis is wrong.** End to end it scores 3/8
on the hard set, against 4-5/8 for the model answering from memory, and every
remaining failure is a wrong *formula* evaluated perfectly:

    time constant of 10k and 100nF  ->  1/(2*pi*R*C)   (that is the cutoff formula)
    300 Hz sampled at 500 Hz        ->  300/(2*pi)     (aliasing is 500-300)
    Shannon capacity                ->  1e6/10*log10(...) mangled

Exact computation of the wrong formula is worse than approximate recall of the
right one, because the answer arrives with more authority and no hedge.

So this is scoped to what it actually helps with: arithmetic the user states
outright ("15 percent of 340", "convert 250mm to inches"). Engineering questions
that need a formula recalled first belong on a bigger model, and that is a
capability gap no tool closes.

Evaluation is a restricted AST walk, not `eval`. The input is model output shaped
by a question Devansh spoke aloud, which is two layers of untrusted, and `eval`
on that is a remote code execution bug waiting for a phrasing that triggers it.
"""

from __future__ import annotations

import ast
import math
import operator
import re

from bruno.tools.base import Privacy, ToolResult, tool

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_NAMES = {
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
    # SI prefixes, because every question here carries units.
    "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12,
    "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
}
_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "min": min, "max": max, "pow": math.pow, "degrees": math.degrees,
    "radians": math.radians, "hypot": math.hypot,
}

MAX_POW = 1024  # guards 9**9**9, which would hang before it ever returned


class CalcError(ValueError):
    pass


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalcError(f"not a number: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise CalcError("unsupported operator")
        left, right = _eval(node.left), _eval(node.right)
        if op is operator.pow and (abs(right) > MAX_POW or abs(left) > MAX_POW):
            raise CalcError("exponent too large")
        if op in (operator.truediv, operator.mod, operator.floordiv) and right == 0:
            raise CalcError("division by zero")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY.get(type(node.op))
        if op is None:
            raise CalcError("unsupported unary operator")
        return op(_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _NAMES:
            return _NAMES[node.id]
        raise CalcError(f"unknown name: {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise CalcError("unknown function")
        if node.keywords:
            raise CalcError("keyword arguments not supported")
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise CalcError(f"unsupported expression: {type(node).__name__}")


def evaluate(expr: str) -> float:
    expr = (expr or "").strip()
    if not expr:
        raise CalcError("empty expression")
    if len(expr) > 500:
        raise CalcError("expression too long")
    # Spoken maths arrives with words attached; strip the common ones.
    expr = re.sub(r"\b(what is|whats|calculate|compute|equals?|answer)\b", "", expr, flags=re.I)
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    expr = expr.replace("Ω", "").replace("ohm", "").replace("Ohm", "")
    # The model writes engineering units the way a person does -- 100nF, 10kΩ,
    # 1uF, 3.3V -- so accept them and keep only the SI prefix as a multiplier.
    # Without this, "10*k * 100*nF" dies on "unknown name: nF" and the whole
    # calculator looks broken when the arithmetic was fine.
    # Order matters. Drop the unit letter that follows an SI prefix first
    # (100nF -> 100n), then a bare unit stuck to a number (3.3V -> 3.3), then
    # turn the surviving prefix into an explicit multiplier (100n -> 100*n).
    # The prefix can be preceded by a digit (100nF) or an explicit operator
    # (100*nF) -- the model writes both, sometimes in the same expression.
    expr = re.sub(r"(?<=[\d*\s(])([kMGTmunpf])(?:F|H|s|V|A|W|Hz|ohms?)\b", r"\1", expr)
    expr = re.sub(r"(?<=\d)(?:F|H|V|A|W|Hz|ohms?)\b", "", expr)
    expr = re.sub(r"(\d)\s*([kMGTmunpf])(?![a-zA-Z0-9.])", r"\1*\2", expr)
    expr = expr.strip(" =?*")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"couldn't parse that: {e.msg}") from e
    return _eval(tree)


def speak_number(x: float) -> str:
    """Format for the ear, not the eye. 0.000159 is unspeakable; 159 microseconds
    is not, and engineering notation is what he'd say out loud anyway."""
    if x == 0:
        return "0"
    if not math.isfinite(x):
        return "undefined"
    exp = math.floor(math.log10(abs(x)) / 3) * 3
    exp = max(-15, min(12, exp))
    prefix = {12: " trillion", 9: " billion", 6: " million", 3: " thousand",
              0: "", -3: " thousandths", -6: " millionths", -9: " billionths",
              -12: " trillionths", -15: " quadrillionths"}[exp]
    mant = x / (10 ** exp)
    return f"{mant:.4g}{prefix}" if exp else f"{mant:.6g}"


@tool(
    name="calculate",
    description=(
        "Evaluate a numeric expression exactly. Use for ANY question requiring "
        "arithmetic, a formula, or a unit conversion — RC time constants, cutoff "
        "frequencies, sample rates, resistor values, percentages. Supply the "
        "expression using digits and operators, with SI prefixes as multipliers "
        "(1*k for kilo, 1*u for micro, 1*n for nano). Never do the arithmetic "
        "yourself; always use this. MATHS ONLY: the request must reduce to an "
        "expression with a numeric answer. A question asking what something IS, "
        "or how two things DIFFER, or why something behaves a certain way, is "
        "never this tool even when it mentions numbers or components."
    ),
    privacy=Privacy.CLOUD_OK,
    arg="expression",
)
def calculate(expression: str, **_) -> ToolResult:
    try:
        value = evaluate(expression)
    except CalcError as e:
        return ToolResult(f"I couldn't work that out: {e}", Privacy.CLOUD_OK, ok=False)
    except (OverflowError, ValueError, ZeroDivisionError) as e:
        return ToolResult(f"That doesn't have a real answer: {e}", Privacy.CLOUD_OK, ok=False)
    return ToolResult(speak_number(value), Privacy.CLOUD_OK,
                      data={"expression": expression, "value": value})
