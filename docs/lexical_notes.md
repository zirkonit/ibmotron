# Lexical Notes

`it_text_v1` is the normalized, line-oriented source format used for generation and training input.

Current token conventions implemented in the renderer:

- `z`: substitution.
- `g`: transfer.
- `t`: output.
- `h`: halt.
- `read`: input.
- `s`: addition.
- `x`: multiplication.
- `m`: unary minus and infix subtraction.
- `d`: division.
- `u`, `w`: relational tokens, matching the shipped SIMH examples.

Historical note:

- `s`, `x`, `u`, `w`, `z`, `g`, `t`, and `h` are directly evidenced by the restored SIMH examples.
- `m` and `d` are implemented from the line-oriented restored notation plus the Carnegie operator representation table; broader corpus validation should keep checking them before large B2/B3 expansion.
- Exponentiation is intentionally excluded from v1 because the restored default P1 package does not provide runtime support.
