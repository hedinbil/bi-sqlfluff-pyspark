"""
AST-based SQL string extraction and replacement functionality.
Uses AST to extract SQL strings from Python code, reformat them with sqlfluff,
and replace them in the original source.
"""

import ast
import io
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from sqlfluff_pyspark.core import fix_sql

logger = logging.getLogger(__name__)

# Placeholder marker for f-string expressions during SQL formatting
FSTRING_PLACEHOLDER = "__FSTRING_EXPR_PLACEHOLDER__"


class SQLStringExtractor(ast.NodeVisitor):
    """AST visitor to extract SQL strings from spark.sql() calls."""

    def __init__(self):
        self.sql_strings: List[Dict[str, any]] = []
        self.current_file: Optional[str] = None

    def visit_Call(self, node: ast.Call):
        """Visit function call nodes to find spark.sql() calls."""
        # Check if this is a spark.sql() call
        if self._is_spark_sql_call(node):
            # Extract string arguments from the call
            for arg in node.args:
                sql_info = self._extract_string_from_node(arg)
                if sql_info is not None:
                    self._record_sql_string(arg, sql_info)
        self.generic_visit(node)

    def _is_spark_sql_call(self, node: ast.Call) -> bool:
        """Check if a call node is a spark.sql() call."""
        # Check if it's an attribute access: spark.sql
        if isinstance(node.func, ast.Attribute):
            # Check if attribute name is "sql"
            if node.func.attr == "sql":
                # Check if the value is "spark" (could be a Name or another Attribute)
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id == "spark"
                elif isinstance(node.func.value, ast.Attribute):
                    # Handle cases like self.spark.sql() or df.spark.sql()
                    # We'll accept any chain ending with spark.sql
                    return self._is_spark_attribute(node.func.value)
        return False

    def _is_spark_attribute(self, node: ast.Attribute) -> bool:
        """Recursively check if an attribute chain ends with 'spark'."""
        if isinstance(node.value, ast.Name):
            return node.value.id == "spark"
        elif isinstance(node.value, ast.Attribute):
            return self._is_spark_attribute(node.value)
        return False

    def _extract_string_from_node(self, node: ast.AST) -> Optional[Union[str, Dict]]:
        """
        Extract string value from an AST node.

        Returns:
            - str: For regular string literals
            - Dict: For f-strings, containing structure information
            - None: If not a string node
        """
        # Handle Python 3.8+ Constant nodes
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value

        # Handle Python <3.8 Str nodes
        if isinstance(node, ast.Str):
            return node.s

        # Handle JoinedStr (f-strings) - extract structure with expressions
        if isinstance(node, ast.JoinedStr):
            return self._extract_fstring_structure(node)

        return None

    def _extract_fstring_structure(self, node: ast.JoinedStr) -> Dict:
        """
        Extract structure from an f-string (JoinedStr) node.

        Returns a dictionary with:
        - 'type': 'fstring'
        - 'parts': List of parts, each being either:
            - {'type': 'str', 'value': str} for static strings
            - {'type': 'expr', 'formatted_value': ast.FormattedValue, 'expr_node': ast.AST} for expressions
        """
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append({"type": "str", "value": value.value})
            elif isinstance(value, ast.Str):
                parts.append({"type": "str", "value": value.s})
            elif isinstance(value, ast.FormattedValue):
                # This is an expression like {table_name}
                # Store both the FormattedValue (for position info) and the expression node
                parts.append(
                    {"type": "expr", "formatted_value": value, "expr_node": value.value}
                )

        if parts:
            return {"type": "fstring", "parts": parts, "node": node}

        return None

    def _record_sql_string(self, node: ast.AST, sql_info: Union[str, Dict]):
        """Record a SQL string found in the AST."""
        if isinstance(sql_info, str):
            # Regular string literal
            self.sql_strings.append(
                {
                    "node": node,
                    "sql": sql_info,
                    "sql_type": "string",
                    "lineno": node.lineno,
                    "col_offset": node.col_offset,
                    "end_lineno": getattr(node, "end_lineno", node.lineno),
                    "col_end_offset": getattr(
                        node, "col_end_offset", node.col_offset + len(sql_info)
                    ),
                }
            )
        elif isinstance(sql_info, dict) and sql_info.get("type") == "fstring":
            # F-string
            self.sql_strings.append(
                {
                    "node": node,
                    "sql": sql_info,
                    "sql_type": "fstring",
                    "lineno": node.lineno,
                    "col_offset": node.col_offset,
                    "end_lineno": getattr(node, "end_lineno", node.lineno),
                    "col_end_offset": getattr(node, "col_end_offset", node.col_offset),
                }
            )


def extract_sql_strings(source_code: str) -> List[Dict[str, any]]:
    """
    Extract SQL strings from spark.sql() calls in Python source code using AST.

    Only extracts SQL strings that are passed as arguments to spark.sql() calls.
    Other SQL strings in the code are ignored.

    Args:
        source_code: Python source code as a string

    Returns:
        List of dictionaries containing SQL string information
    """
    try:
        tree = ast.parse(source_code)
        extractor = SQLStringExtractor()
        extractor.visit(tree)
        return extractor.sql_strings
    except SyntaxError as e:
        logger.error(f"Failed to parse Python code: {e}")
        return []


def _extract_expression_source(
    source_code: str, expr_node: ast.AST, formatted_value: ast.FormattedValue = None
) -> str:
    """Extract the source code for an expression node from the original source.

    For f-strings like f"SELECT * FROM {table_name}", this extracts "table_name"
    (the expression inside the braces, without the braces themselves).

    Args:
        source_code: Original source code
        expr_node: The AST expression node (e.g., Name, Attribute, etc.)
        formatted_value: Optional FormattedValue node to help locate braces in source
    """
    # Try using ast.get_source_segment if available (Python 3.8+)
    try:
        if hasattr(ast, "get_source_segment"):
            # Use the FormattedValue position to find braces, or expr_node if no FormattedValue
            node_to_use = formatted_value if formatted_value is not None else expr_node
            segment = ast.get_source_segment(source_code, node_to_use)
            if segment:
                # If we got the segment from FormattedValue, it might include braces
                # Remove braces if present
                segment = segment.strip()
                if segment.startswith("{") and segment.endswith("}"):
                    segment = segment[1:-1]
                return segment
    except (AttributeError, TypeError):
        pass

    # Fallback: Use FormattedValue position to find braces in source
    if formatted_value is not None:
        lines = source_code.splitlines(keepends=True)
        lineno = formatted_value.lineno - 1  # Convert to 0-based
        end_lineno = getattr(formatted_value, "end_lineno", formatted_value.lineno) - 1

        if lineno < len(lines):
            # Find the opening brace before the expression
            line = lines[lineno]
            start_col = formatted_value.col_offset

            # Look backwards for opening brace
            brace_start = start_col
            while brace_start > 0 and line[brace_start - 1] != "{":
                brace_start -= 1

            # Find the closing brace after the expression
            if end_lineno < len(lines):
                end_line = lines[end_lineno]
                end_col = getattr(formatted_value, "col_end_offset", start_col)

                # Look forwards for closing brace
                brace_end = end_col
                while brace_end < len(end_line) and end_line[brace_end] != "}":
                    brace_end += 1

                # Extract content between braces
                if brace_start < start_col and brace_end <= len(end_line):
                    if lineno == end_lineno:
                        return line[brace_start + 1 : brace_end].strip()
                    else:
                        # Multi-line: combine lines
                        result = line[brace_start + 1 :]
                        for i in range(lineno + 1, end_lineno):
                            if i < len(lines):
                                result += lines[i]
                        if end_lineno < len(lines):
                            result += end_line[:brace_end]
                        return result.strip()

    # Final fallback: Use expr_node position directly
    lineno = expr_node.lineno - 1  # Convert to 0-based
    end_lineno = getattr(expr_node, "end_lineno", expr_node.lineno) - 1

    lines = source_code.splitlines(keepends=True)

    if lineno < len(lines):
        if lineno == end_lineno:
            # Single line expression
            line = lines[lineno]
            start_col = expr_node.col_offset
            end_col = getattr(expr_node, "col_end_offset", expr_node.col_offset)
            return line[start_col:end_col].rstrip()
        else:
            # Multi-line expression
            result = []
            for i in range(lineno, end_lineno + 1):
                if i < len(lines):
                    if i == lineno:
                        # First line - start from col_offset
                        result.append(lines[i][expr_node.col_offset :])
                    elif i == end_lineno:
                        # Last line - end at col_end_offset
                        end_col = getattr(expr_node, "col_end_offset", len(lines[i]))
                        result.append(lines[i][:end_col])
                    else:
                        # Middle lines - take entire line
                        result.append(lines[i])
            return "".join(result).rstrip()

    return ""


def _format_fstring_sql(
    fstring_info: Dict,
    source_code: str,
    config_path: str,
    quote_style: str = '"',
) -> Optional[str]:
    """
    Format SQL in an f-string while preserving expressions.

    Uses placeholders to combine all SQL parts, format as one unit,
    then split back and reconstruct the f-string.

    Args:
        fstring_info: Dictionary with 'parts' list from _extract_fstring_structure
        source_code: Original source code to extract expression strings
        config_path: Path to sqlfluff config
        quote_style: Quote style to use (single, double, or triple quotes)

    Returns:
        Formatted f-string string with quotes, or None if formatting failed
    """
    parts = fstring_info["parts"]

    # Build SQL with unique placeholders for expressions
    expressions = []
    sql_with_placeholders = []
    placeholder_patterns = []

    for i, part in enumerate(parts):
        if part["type"] == "str":
            sql_with_placeholders.append(part["value"])
        elif part["type"] == "expr":
            # Extract expression source from the expression node, using FormattedValue for position
            formatted_value = part.get("formatted_value")
            expr_source = _extract_expression_source(
                source_code, part["expr_node"], formatted_value
            )
            expressions.append(expr_source)
            # Create a unique placeholder that won't appear in SQL
            placeholder = f"{FSTRING_PLACEHOLDER}{i:04d}{FSTRING_PLACEHOLDER}"
            placeholder_patterns.append((placeholder, expr_source))
            sql_with_placeholders.append(placeholder)

    # Combine and format the SQL
    combined_sql = "".join(sql_with_placeholders)

    if not combined_sql.strip():
        return None

    try:
        formatted_sql = fix_sql(combined_sql, config_path)
    except Exception as e:
        logger.warning(f"Failed to format f-string SQL: {e}")
        return None

    # Replace placeholders back with expressions
    # Process in reverse order to avoid issues with overlapping patterns
    # Note: sqlfluff may change the case of the placeholder, so we need to do
    # case-insensitive replacement
    result = formatted_sql
    for placeholder, expr_source in reversed(placeholder_patterns):
        # Use regex for case-insensitive replacement since sqlfluff may lowercase placeholders
        pattern = re.escape(placeholder)
        result = re.sub(pattern, f"{{{expr_source}}}", result, flags=re.IGNORECASE)

    # Determine if we should use triple quotes
    use_triple_quotes = "\n" in result or len(result) > 80

    if use_triple_quotes and quote_style in ['"', "'"]:
        quote_style = quote_style * 3

    # Reconstruct the f-string with 'f' prefix and quotes
    return f"f{quote_style}{result}{quote_style}"


def _get_fstring_bounds_and_quote(
    source_code: str, node: ast.JoinedStr
) -> Tuple[int, int, int, int, str]:
    """
    Get the start and end positions of an f-string in the source code, and its quote style.

    Returns:
        (start_line, start_col, end_line, end_col, quote_style) tuple (0-based)
    """
    start_line = node.lineno - 1
    start_col = node.col_offset
    end_line = getattr(node, "end_lineno", node.lineno) - 1
    end_col = getattr(node, "col_end_offset", node.col_offset)

    lines = source_code.splitlines(keepends=True)

    # Find the actual f-string bounds including the 'f' prefix and quotes
    # Look backwards from start_col to find 'f' and quote
    if start_line < len(lines):
        line = lines[start_line]

        # Look for 'f' prefix (could be before start_col)
        f_pos = start_col
        while f_pos > 0 and line[f_pos - 1] in "fF":
            f_pos -= 1

        # Look for quote style
        quote_chars = ['"""', "'''", '"', "'"]  # noqa: E501
        quote_style = None
        quote_start = None

        # Check if there's an 'f' prefix
        has_f_prefix = f_pos < start_col and line[f_pos] in "fF"
        search_start = f_pos + (1 if has_f_prefix else 0)

        for quote in quote_chars:
            if search_start + len(quote) <= len(line):
                if line[search_start : search_start + len(quote)] == quote:
                    quote_style = quote
                    quote_start = search_start
                    break

        if quote_style:
            # Find closing quote
            search_start = quote_start + len(quote_style)

            if quote_style in ['"""', "'''"]:
                # Multi-line string
                current_line = start_line
                search_pos = search_start if current_line == start_line else 0

                while current_line < len(lines):
                    line_text = lines[current_line]
                    pos = line_text.find(quote_style, search_pos)
                    if pos != -1:
                        quote_end = pos + len(quote_style)
                        return (start_line, f_pos, current_line, quote_end, quote_style)
                    current_line += 1
                    search_pos = 0
            else:
                # Single-line string
                pos = line.find(quote_style, search_start)
                if pos != -1:
                    quote_end = pos + len(quote_style)
                    return (start_line, f_pos, start_line, quote_end, quote_style)

    # Fallback to AST positions with default quote style
    return (start_line, start_col, end_line, end_col, '"')


def replace_sql_in_source(
    source_code: str,
    sql_replacements: List[Tuple[int, int, int, int, str]],
) -> str:
    """
    Replace SQL strings in source code with reformatted versions using line/column positions.

    Args:
        source_code: Original Python source code
        sql_replacements: List of (start_line, start_col, end_line, end_col, new_sql) tuples

    Returns:
        Modified source code with SQL strings replaced
    """
    # Sort replacements by line number (reverse order to maintain indices)
    sql_replacements.sort(key=lambda x: (x[0], x[1]), reverse=True)

    lines = source_code.splitlines(keepends=True)
    if not lines:
        return source_code

    # Ensure all lines end with newline for proper reconstruction
    for i, line in enumerate(lines):
        if not line.endswith(("\n", "\r\n", "\r")):
            lines[i] = line + "\n"

    for start_line, start_col, end_line, end_col, new_sql in sql_replacements:
        if start_line == end_line:
            # Single line replacement
            line = lines[start_line]
            # Handle case where line might not have newline
            line_content = line.rstrip("\n\r")
            newline = line[len(line_content) :]
            lines[start_line] = (
                line_content[:start_col] + new_sql + line_content[end_col:] + newline
            )
        else:
            # Multi-line replacement
            start_line_content = lines[start_line][:start_col]
            end_line_content = lines[end_line][end_col:]
            lines[start_line] = start_line_content + new_sql + end_line_content
            # Remove lines in between and the end line (since its content is now in start_line)
            for i in range(start_line + 1, end_line + 1):
                lines[i] = ""

    return "".join(lines)


def reformat_sql_in_python_file(
    file_path: str,
    config_path: str,
    dry_run: bool = False,
) -> Dict[str, any]:
    """
    Extract SQL strings from spark.sql() calls in a Python file, reformat them, and replace them.

    Only processes SQL strings found in spark.sql() calls. Uses AST to extract SQL strings,
    writes them to temporary StringIO objects for processing, then replaces them in the original source.

    Args:
        file_path: Path to Python file to process
        config_path: Path to sqlfluff config file
        dry_run: If True, don't write changes back to file

    Returns:
        Dictionary with results including replacements made
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Python file not found: {file_path}")

    # Read source code
    source_code = file_path_obj.read_text(encoding="utf-8")

    # Extract SQL strings using AST
    sql_strings = extract_sql_strings(source_code)

    if not sql_strings:
        logger.info(f"No SQL strings found in {file_path}")
        return {
            "file": file_path,
            "sql_strings_found": 0,
            "sql_strings_reformatted": 0,
            "replacements": [],
        }

    logger.info(f"Found {len(sql_strings)} SQL string(s) in {file_path}")

    # Reformat each SQL string
    replacements = []
    reformatted_count = 0

    for sql_info in sql_strings:
        sql_type = sql_info.get("sql_type", "string")

        try:
            if sql_type == "fstring":
                # Handle f-string formatting
                fstring_info = sql_info["sql"]
                node = sql_info["node"]

                # Get the bounds and quote style of the f-string in source
                start_line, start_col, end_line, end_col, quote_style = (
                    _get_fstring_bounds_and_quote(source_code, node)
                )

                # Format the f-string SQL
                formatted_fstring = _format_fstring_sql(
                    fstring_info, source_code, config_path, quote_style
                )

                if formatted_fstring is not None:
                    # Extract the original f-string to compare
                    lines = source_code.splitlines(keepends=True)
                    original_fstring = ""
                    if start_line == end_line:
                        line = lines[start_line]
                        original_fstring = line[start_col:end_col].rstrip()
                    else:
                        # Multi-line
                        for i in range(start_line, end_line + 1):
                            if i < len(lines):
                                if i == start_line:
                                    original_fstring += lines[i][start_col:]
                                elif i == end_line:
                                    original_fstring += lines[i][:end_col]
                                else:
                                    original_fstring += lines[i]
                        original_fstring = original_fstring.rstrip()

                    # Only replace if it changed
                    if formatted_fstring.strip() != original_fstring.strip():
                        replacements.append(
                            (
                                start_line,
                                start_col,
                                end_line,
                                end_col,
                                formatted_fstring,
                            )
                        )
                        reformatted_count += 1
                        logger.info(
                            f"Reformatted f-string SQL at line {sql_info['lineno']}"
                        )
            else:
                # Handle regular string literal
                original_sql = sql_info["sql"]

                # Use StringIO as a "partial file" to hold the SQL string
                sql_buffer = io.StringIO(original_sql)
                sql_content = sql_buffer.read()
                sql_buffer.close()

                # Fix the SQL using sqlfluff
                fixed_sql = fix_sql(sql_content, config_path)

                # Only replace if it changed
                if fixed_sql.strip() != original_sql.strip():
                    # Get line/column positions from AST node
                    # The col_offset points to the start of the string literal (including opening quote)
                    start_line = sql_info["lineno"] - 1  # Convert to 0-based
                    start_col = sql_info["col_offset"]
                    end_line = sql_info["end_lineno"] - 1  # Convert to 0-based
                    end_col = sql_info["col_end_offset"]

                    # Find the actual string literal in source to get exact positions and quote style
                    lines = source_code.splitlines(keepends=True)
                    if start_line < len(lines):
                        line = lines[start_line]

                        # Find the quote style and full string literal
                        quote_chars = [
                            '"""',
                            "'''",
                            '"',
                            "'",
                        ]  # Check triple quotes first
                        quote_style = None
                        quote_start = None

                        for quote in quote_chars:
                            if start_col + len(quote) <= len(line):
                                if line[start_col : start_col + len(quote)] == quote:
                                    quote_style = quote
                                    quote_start = start_col
                                    break

                        if quote_style:
                            # Find the closing quote position
                            # If AST provides col_end_offset, use it (but verify it's correct)
                            # Otherwise, search for the closing quote
                            quote_end = None

                            # Check if we have a valid AST end position
                            if end_col is not None and end_line < len(lines):
                                end_line_content = lines[end_line]
                                # Verify the AST position has the closing quote
                                if end_col >= len(quote_style) and end_col <= len(
                                    end_line_content
                                ):
                                    actual_end_quote = end_line_content[
                                        end_col - len(quote_style) : end_col
                                    ]
                                    if actual_end_quote == quote_style:
                                        # AST position is correct
                                        quote_end = end_col

                            # If AST position is not available or incorrect, search for the quote
                            if quote_end is None:
                                search_start = quote_start + len(quote_style)

                                if quote_style in ['"""', "'''"]:
                                    # Multi-line string: search from start_line to end_line
                                    current_line = start_line
                                    search_pos = (
                                        search_start
                                        if current_line == start_line
                                        else 0
                                    )

                                    # Limit search to end_line to avoid finding wrong quotes
                                    while (
                                        current_line <= end_line
                                        and current_line < len(lines)
                                    ):
                                        line_text = lines[current_line]
                                        # On the end_line, only search up to a reasonable limit
                                        # to avoid going past the actual string
                                        search_limit = len(line_text)
                                        if (
                                            current_line == end_line
                                            and end_col is not None
                                        ):
                                            # If we have an AST end hint, search only up to that + some margin
                                            search_limit = min(
                                                end_col + 20, len(line_text)
                                            )

                                        pos = line_text.find(
                                            quote_style, search_pos, search_limit
                                        )
                                        if pos != -1:
                                            quote_end = pos + len(quote_style)
                                            end_line = current_line
                                            break
                                        current_line += 1
                                        search_pos = 0
                                else:
                                    # Single-line string: search on the same line
                                    # Limit search to avoid finding quotes from other strings
                                    search_limit = len(line)
                                    if end_col is not None:
                                        # Use AST hint if available
                                        search_limit = min(end_col + 20, len(line))
                                    pos = line.find(
                                        quote_style, search_start, search_limit
                                    )
                                    if pos != -1:
                                        quote_end = pos + len(quote_style)
                                        end_line = start_line

                            if quote_end is not None:
                                # Determine if we should use triple quotes for the fixed SQL
                                use_triple_quotes = (
                                    "\n" in fixed_sql or len(fixed_sql) > 80
                                )

                                if use_triple_quotes and quote_style in ['"', "'"]:
                                    # Switch to triple quotes
                                    new_quote_style = quote_style * 3
                                    fixed_sql_quoted = (
                                        f"{new_quote_style}{fixed_sql}{new_quote_style}"
                                    )
                                else:
                                    # Keep original quote style
                                    fixed_sql_quoted = (
                                        f"{quote_style}{fixed_sql}{quote_style}"
                                    )

                                # Replace the entire string literal (including quotes)
                                replacements.append(
                                    (
                                        start_line,
                                        quote_start,
                                        end_line,
                                        quote_end,
                                        fixed_sql_quoted,
                                    )
                                )
                                reformatted_count += 1
                                logger.info(
                                    f"Reformatted SQL string at line {sql_info['lineno']}"
                                )
                            else:
                                logger.warning(
                                    f"Could not find closing quote for SQL at line {sql_info['lineno']}"
                                )
                        else:
                            logger.warning(
                                f"Could not determine quote style for SQL at line {sql_info['lineno']}"
                            )
        except Exception as e:
            logger.warning(f"Failed to reformat SQL at line {sql_info['lineno']}: {e}")

    # Apply replacements
    modified_source = source_code
    if replacements:
        modified_source = replace_sql_in_source(source_code, replacements)
        if not dry_run:
            file_path_obj.write_text(modified_source, encoding="utf-8")
            logger.info(f"Wrote reformatted code to {file_path}")
        else:
            logger.info(f"Dry run: would write reformatted code to {file_path}")

    return {
        "file": file_path,
        "sql_strings_found": len(sql_strings),
        "sql_strings_reformatted": reformatted_count,
        "replacements": replacements,
        "dry_run": dry_run,
        "modified_source": modified_source if dry_run else None,
    }
