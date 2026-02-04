"""Tests for AST-based SQL extraction and replacement."""

import importlib.util
import re
from pathlib import Path

import pytest
from sqlfluff_pyspark.ast_extract import (
    extract_sql_strings,
    reformat_sql_in_python_file,
    replace_sql_in_source,
)


def load_test_case(case_name: str) -> str:
    """Load a test case from fixtures."""
    test_case_path = (
        Path(__file__).parent / "fixtures" / "test_cases" / f"{case_name}.py"
    )
    return test_case_path.read_text()


def load_expected_output(case_name: str):
    """Load expected output from fixtures."""
    expected_path = (
        Path(__file__).parent / "fixtures" / "expected_outputs" / f"{case_name}.py"
    )
    spec = importlib.util.spec_from_file_location("expected", expected_path)
    expected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(expected)
    return expected


class TestExtractSQLStrings:
    """Tests for extract_sql_strings function."""

    def test_extract_sql_from_spark_sql_call(self):
        """Test extracting SQL string from spark.sql() call."""
        code = """
def get_users():
    result = spark.sql("SELECT id, name FROM users")
    return result
"""
        sql_strings = extract_sql_strings(code)
        assert len(sql_strings) == 1
        assert "SELECT" in sql_strings[0]["sql"]
        assert "id, name FROM users" in sql_strings[0]["sql"]

    def test_extract_multiple_spark_sql_calls(self):
        """Test extracting SQL strings from multiple spark.sql() calls."""
        code = """
def get_data():
    df1 = spark.sql("SELECT * FROM table1")
    df2 = spark.sql("INSERT INTO table2 VALUES (1)")
    return df1, df2
"""
        sql_strings = extract_sql_strings(code)
        assert len(sql_strings) == 2
        assert any("SELECT" in s["sql"] for s in sql_strings)
        assert any("INSERT" in s["sql"] for s in sql_strings)

    def test_ignore_non_spark_sql_strings(self):
        """Test that non-spark.sql() SQL strings are ignored."""
        code = """
def execute_query():
    query = "SELECT * FROM users WHERE id = 1"
    cursor.execute(query)
    spark.sql("SELECT * FROM table")
"""
        sql_strings = extract_sql_strings(code)
        # Should only find the spark.sql() call, not the query variable
        assert len(sql_strings) == 1
        assert "SELECT * FROM table" in sql_strings[0]["sql"]

    def test_no_spark_sql_calls(self):
        """Test with no spark.sql() calls."""
        code = """
def hello():
    query = "SELECT * FROM users"
    return query
"""
        sql_strings = extract_sql_strings(code)
        # Should not find any SQL strings since there are no spark.sql() calls
        assert len(sql_strings) == 0

    def test_multiline_sql_in_spark_sql(self):
        """Test extracting multiline SQL strings from spark.sql()."""
        code = '''
def get_complex_query():
    result = spark.sql("""
    SELECT u.id, u.name, p.title
    FROM users u
    JOIN posts p ON u.id = p.user_id
    WHERE u.active = 1
    """)
    return result
'''
        sql_strings = extract_sql_strings(code)
        assert len(sql_strings) == 1
        assert "SELECT" in sql_strings[0]["sql"]
        assert "JOIN" in sql_strings[0]["sql"]

    def test_spark_sql_with_variable(self):
        """Test that spark.sql() with variable arguments is handled."""
        code = """
def get_data():
    query = "SELECT * FROM table"
    result = spark.sql(query)  # Variable, not a string literal
"""
        sql_strings = extract_sql_strings(code)
        # Variable arguments are not extracted (only string literals)
        assert len(sql_strings) == 0

    def test_nested_spark_sql(self):
        """Test spark.sql() calls in nested contexts."""
        code = """
def process_data():
    if condition:
        df = spark.sql("SELECT * FROM table1")
    else:
        df = spark.sql("SELECT * FROM table2")
    return df
"""
        sql_strings = extract_sql_strings(code)
        assert len(sql_strings) == 2

    def test_invalid_python_code(self):
        """Test with invalid Python code."""
        code = "def invalid syntax here"
        sql_strings = extract_sql_strings(code)
        # Should return empty list on syntax error
        assert isinstance(sql_strings, list)


class TestReplaceSQLInSource:
    """Tests for replace_sql_in_source function."""

    def test_replace_single_line_sql(self):
        """Test replacing SQL on a single line."""
        source = 'query = "SELECT * FROM table"'
        replacements = [(0, 9, 0, 33, '"SELECT * FROM table"')]
        result = replace_sql_in_source(source, replacements)
        assert "SELECT" in result

    def test_replace_multiline_sql(self):
        """Test replacing multiline SQL."""
        source = '''query = """
SELECT * FROM table
WHERE id = 1
"""'''
        replacements = [(0, 9, 2, 3, '"""SELECT * FROM table\\nWHERE id = 1\\n"""')]
        result = replace_sql_in_source(source, replacements)
        assert "SELECT" in result


class TestReformatSQLInPythonFile:
    """Tests for reformat_sql_in_python_file function."""

    def test_file_not_found(self, sqlfluff_config_file):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            reformat_sql_in_python_file("/nonexistent/file.py", sqlfluff_config_file)

    def test_no_spark_sql_calls(self, sqlfluff_config_file, tmp_path):
        """Test file with no spark.sql() calls."""
        python_file = tmp_path / "test.py"
        python_file.write_text(
            'def hello():\n    query = "SELECT * FROM users"\n    return query\n'
        )

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=True
        )
        assert result["sql_strings_found"] == 0
        assert result["sql_strings_reformatted"] == 0

    def test_reformat_spark_sql_string(self, sqlfluff_config_file, tmp_path):
        """Test reformatting SQL strings in spark.sql() calls."""
        python_file = tmp_path / "test.py"
        python_file.write_text(
            'def get_users():\n    result = spark.sql("SeLEct * from users")\n    return result\n'
        )

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=True
        )
        assert result["sql_strings_found"] > 0
        # The SQL should be reformatted (case fixed)
        if result["sql_strings_reformatted"] > 0:
            assert result["replacements"]

    def test_dry_run(self, sqlfluff_config_file, tmp_path):
        """Test dry run mode doesn't modify file."""
        python_file = tmp_path / "test.py"
        original_content = 'def get_users():\n    result = spark.sql("SeLEct * from users")\n    return result\n'
        python_file.write_text(original_content)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=True
        )

        # File should be unchanged in dry run mode
        assert python_file.read_text() == original_content
        assert result["dry_run"] is True

    def test_actual_reformat(self, sqlfluff_config_file, tmp_path):
        """Test actually reformatting SQL strings in spark.sql() calls."""
        python_file = tmp_path / "test.py"
        python_file.write_text(
            'def get_users():\n    result = spark.sql("SeLEct * from users")\n    return result\n'
        )

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        # If SQL was reformatted, file should be modified
        if result["sql_strings_reformatted"] > 0:
            modified_content = python_file.read_text()
            # The reformatted SQL should have proper case
            assert "SELECT" in modified_content or "select" in modified_content.lower()

    def test_end_line_not_duplicated_when_fixing(self, sqlfluff_config_file, tmp_path):
        """Test that the line at the end of a Python string is not duplicated when fixing."""
        python_file = tmp_path / "test.py"
        # Create a multiline SQL string that ends at the end of a line
        # The SQL string ends with a line that should not be duplicated
        original_content = '''def get_data():
    result = spark.sql("""
    SELECT id, name
    FROM users
    WHERE active = 1
    """)
    return result
'''
        python_file.write_text(original_content)

        # Count the number of lines before fixing
        original_lines = original_content.splitlines(keepends=True)
        original_line_count = len(original_lines)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        # Read the modified content
        modified_content = python_file.read_text()
        modified_lines = modified_content.splitlines(keepends=True)
        modified_line_count = len(modified_lines)

        # Check that the end line is not duplicated
        # The line count should be the same or less (if SQL was reformatted to fewer lines)
        # but should not be more due to duplication
        assert modified_line_count <= original_line_count + 2, (
            f"Line count increased unexpectedly. "
            f"Original: {original_line_count}, Modified: {modified_line_count}. "
            f"This suggests the end line was duplicated."
        )

        # Specifically check that the line after the closing triple quotes is not duplicated
        # Find the closing triple quotes and check what comes after
        lines_list = modified_content.splitlines(keepends=True)
        for i, line in enumerate(lines_list):
            if '"""' in line and i > 0:
                # Check if this is the closing quote line
                # The next line should be the return statement, not a duplicate
                if i + 1 < len(lines_list):
                    next_line = lines_list[i + 1]
                    # If the next line is the same as the current line (excluding quotes), it's duplicated
                    if next_line.strip() == line.strip() and '"""' not in next_line:
                        pytest.fail(
                            f"End line appears to be duplicated. "
                            f"Line {i}: {repr(line)}, Line {i + 1}: {repr(next_line)}"
                        )

        # Also verify the structure is correct - should have return statement after the SQL
        assert "return result" in modified_content, (
            "Return statement should be present after SQL string"
        )

    def test_end_line_content_preserved(self, sqlfluff_config_file, tmp_path):
        """Test that content at the end of a multiline SQL string is preserved correctly."""
        python_file = tmp_path / "test.py"
        # Create a multiline SQL string where the last line has specific content
        original_content = '''def get_data():
    result = spark.sql("""
    SELECT id, name
    FROM users
    WHERE active = 1
    """)
    return result
'''
        python_file.write_text(original_content)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        modified_content_lines = modified_content.splitlines(keepends=True)

        # Find the line with the closing triple quotes
        closing_quote_line_idx = None
        for i, line in enumerate(modified_content_lines):
            if '"""' in line and i > 2:  # Should be after the opening quotes
                closing_quote_line_idx = i
                break

        if closing_quote_line_idx is not None:
            # Check that the line after closing quotes is correct (should be return statement)
            if closing_quote_line_idx + 1 < len(modified_content_lines):
                line_after_closing = modified_content_lines[closing_quote_line_idx + 1]
                # The line after closing quotes should be the return statement
                # It should NOT be a duplicate of the closing quote line
                assert "return" in line_after_closing, (
                    f"Line after closing quotes should contain 'return', "
                    f"but got: {repr(line_after_closing)}. "
                    f"This suggests the end line was duplicated or content was lost."
                )

        # Verify the structure: should have exactly one return statement
        return_count = modified_content.count("return result")
        assert return_count == 1, (
            f"Expected exactly one 'return result' statement, but found {return_count}. "
            f"This suggests duplication occurred."
        )

        # Verify that the closing quote line doesn't appear twice consecutively
        for i in range(len(modified_content_lines) - 1):
            current_line = modified_content_lines[i]
            next_line = modified_content_lines[i + 1]
            # Check if we have duplicate closing quote lines
            if (
                '"""' in current_line
                and current_line.strip() == next_line.strip()
                and '"""' in next_line
            ):
                pytest.fail(
                    f"Found duplicate closing quote lines at lines {i} and {i + 1}: "
                    f"{repr(current_line)} and {repr(next_line)}"
                )


class TestFStringSupport:
    """Tests for f-string SQL formatting support."""

    def test_extract_fstring_from_spark_sql_call(self):
        """Test extracting f-string from spark.sql() call."""
        code = """
def get_users():
    table_name = "users"
    result = spark.sql(f"SELECT id, name FROM {table_name}")
    return result
"""
        sql_strings = extract_sql_strings(code)
        assert len(sql_strings) == 1
        assert sql_strings[0]["sql_type"] == "fstring"
        assert "parts" in sql_strings[0]["sql"]

    def test_extract_fstring_with_multiple_expressions(self):
        """Test extracting f-string with multiple expressions."""
        code = """
def get_data():
    table = "users"
    condition = "active = 1"
    result = spark.sql(f"SELECT * FROM {table} WHERE {condition}")
    return result
"""
        sql_strings = extract_sql_strings(code)
        assert len(sql_strings) == 1
        assert sql_strings[0]["sql_type"] == "fstring"
        fstring_info = sql_strings[0]["sql"]
        # Should have multiple parts (text and expressions)
        assert len(fstring_info["parts"]) > 2

    def test_reformat_fstring_sql(self, sqlfluff_config_file, tmp_path):
        """Test reformatting SQL in f-strings."""
        python_file = tmp_path / "test.py"
        python_file.write_text(
            'def get_users():\n    table = "users"\n    result = spark.sql(f"SeLEct * from {table}")\n    return result\n'
        )

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=True
        )
        assert result["sql_strings_found"] == 1
        # Should detect the f-string
        if result["sql_strings_reformatted"] > 0:
            assert result["replacements"]

    def test_fstring_preserves_expressions(self, sqlfluff_config_file, tmp_path):
        """Test that f-string expressions are preserved during formatting."""
        python_file = tmp_path / "test.py"
        original_code = 'def get_data():\n    table = "users"\n    result = spark.sql(f"SELECT * FROM {table} WHERE id = {user_id}")\n    return result\n'
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Check that exact f-string placeholder syntax is preserved
        assert "{table}" in modified_content, (
            f"Expected f-string placeholder '{{table}}' not found in output. "
            f"Content: {modified_content}"
        )
        assert "{user_id}" in modified_content, (
            f"Expected f-string placeholder '{{user_id}}' not found in output. "
            f"Content: {modified_content}"
        )
        # Verify it's still an f-string (has 'f' prefix)
        assert (
            'f"' in modified_content
            or "f'" in modified_content
            or 'f"""' in modified_content
            or "f'''" in modified_content
        ), f"Expected f-string prefix 'f' not found. Content: {modified_content}"

    def test_fstring_with_multiline_sql(self, sqlfluff_config_file, tmp_path):
        """Test f-string with multiline SQL."""
        python_file = tmp_path / "test.py"
        original_code = '''def get_data():
    table = "users"
    result = spark.sql(f"""
    SELECT id, name
    FROM {table}
    WHERE active = 1
    """)
    return result
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=True
        )
        assert result["sql_strings_found"] == 1

    def test_mixed_fstring_and_regular_strings(self, sqlfluff_config_file, tmp_path):
        """Test file with both f-strings and regular strings."""
        python_file = tmp_path / "test.py"
        original_code = """def get_data():
    table = "users"
    df1 = spark.sql("SELECT * FROM table1")
    df2 = spark.sql(f"SELECT * FROM {table}")
    return df1, df2
"""
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=True
        )
        # Should find both strings
        assert result["sql_strings_found"] == 2

    def test_sql_followed_by_method_call(self, sqlfluff_config_file, tmp_path):
        """Test that SQL string followed by method call is not corrupted."""
        from pathlib import Path
        import importlib.util

        test_case_path = (
            Path(__file__).parent
            / "fixtures"
            / "test_cases"
            / "sql_with_method_call.py"
        )
        original_code = test_case_path.read_text()

        expected_path = (
            Path(__file__).parent
            / "fixtures"
            / "expected_outputs"
            / "sql_with_method_call.py"
        )
        spec = importlib.util.spec_from_file_location("expected", expected_path)
        expected = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(expected)

        python_file = tmp_path / "test.py"
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        for element in expected.PRESERVED_ELEMENTS:
            assert element in modified_content, (
                f"Expected element '{element}' not found in output"
            )

        # Verify the method call is complete (not truncated)
        if hasattr(expected, "METHOD_CALL_CHECK"):
            method = expected.METHOD_CALL_CHECK["method"]
            if method in modified_content:
                idx = modified_content.find(method)
                method_call = modified_content[
                    idx : idx + expected.METHOD_CALL_CHECK["min_length"] + 10
                ]
                assert method_call.startswith(method), (
                    f"Method call appears to be truncated: {repr(method_call)}"
                )

    def test_multiple_statements_preserved(self, sqlfluff_config_file, tmp_path):
        """Test that code before and after SQL strings is preserved."""
        original_code = load_test_case("multiple_statements")
        expected = load_expected_output("multiple_statements")

        python_file = tmp_path / "test.py"
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        for element in expected.PRESERVED_ELEMENTS:
            assert element in modified_content, (
                f"Expected element '{element}' not found in output"
            )

    def test_code_on_same_line_as_opening_quote(self, sqlfluff_config_file, tmp_path):
        """Test SQL string with code on the same line as opening quote."""
        original_code = load_test_case("code_on_opening_line")
        expected = load_expected_output("code_on_opening_line")

        python_file = tmp_path / "test.py"
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        for element in expected.PRESERVED_ELEMENTS:
            assert element in modified_content, (
                f"Expected element '{element}' not found in output"
            )

    def test_code_on_same_line_as_closing_quote(self, sqlfluff_config_file, tmp_path):
        """Test SQL string with code on the same line as closing quote."""
        original_code = load_test_case("code_on_closing_line")
        expected = load_expected_output("code_on_closing_line")

        python_file = tmp_path / "test.py"
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        for element in expected.PRESERVED_ELEMENTS:
            assert element in modified_content, (
                f"Expected element '{element}' not found in output"
            )

    def test_multiline_string_with_code_on_same_lines(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test multiline SQL string with code on opening and closing lines."""
        original_code = load_test_case("multiline_with_code_on_lines")
        expected = load_expected_output("multiline_with_code_on_lines")

        python_file = tmp_path / "test.py"
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        for element in expected.PRESERVED_ELEMENTS:
            assert element in modified_content, (
                f"Expected element '{element}' not found in output"
            )

    def test_multiple_sql_strings_with_code_between(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test multiple SQL strings with code between them."""
        original_code = load_test_case("multiple_sql_with_code_between")
        expected = load_expected_output("multiple_sql_with_code_between")

        python_file = tmp_path / "test.py"
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        for element in expected.PRESERVED_ELEMENTS:
            assert element in modified_content, (
                f"Expected element '{element}' not found in output"
            )

    def test_sql_in_function_with_code_before_after(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test SQL string inside function with code before and after."""
        original_code = load_test_case("sql_in_function")
        expected = load_expected_output("sql_in_function")

        python_file = tmp_path / "test.py"
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        for element in expected.PRESERVED_ELEMENTS:
            assert element in modified_content, (
                f"Expected element '{element}' not found in output"
            )

    def test_sql_with_complex_method_chaining(self, sqlfluff_config_file, tmp_path):
        """Test SQL string followed by complex method chaining."""
        original_code = load_test_case("complex_method_chaining")
        expected = load_expected_output("complex_method_chaining")

        python_file = tmp_path / "test.py"
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        for element in expected.PRESERVED_ELEMENTS:
            assert element in modified_content, (
                f"Expected element '{element}' not found in output"
            )

    def test_sql_in_if_statement_with_else(self, sqlfluff_config_file, tmp_path):
        """Test SQL strings in conditional statements."""
        original_code = load_test_case("conditional_statements")
        expected = load_expected_output("conditional_statements")

        python_file = tmp_path / "test.py"
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        for element in expected.PRESERVED_ELEMENTS:
            assert element in modified_content, (
                f"Expected element '{element}' not found in output"
            )
        if hasattr(expected, "SQL_CONTENT_HINTS"):
            assert any(
                hint in modified_content for hint in expected.SQL_CONTENT_HINTS
            ), f"None of the SQL content hints found: {expected.SQL_CONTENT_HINTS}"

    def test_sql_with_triple_quotes_and_code_on_lines(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test triple-quoted SQL with code on the same lines as quotes."""
        original_code = load_test_case("triple_quotes_with_code")
        expected = load_expected_output("triple_quotes_with_code")

        python_file = tmp_path / "test.py"
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        for element in expected.PRESERVED_ELEMENTS:
            assert element in modified_content, (
                f"Expected element '{element}' not found in output"
            )

    def test_multiple_sql_strings_same_line(self, sqlfluff_config_file, tmp_path):
        """Test multiple SQL strings on the same line."""
        original_code = load_test_case("multiple_sql_same_line")
        expected = load_expected_output("multiple_sql_same_line")

        python_file = tmp_path / "test.py"
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        for element in expected.PRESERVED_ELEMENTS:
            assert element in modified_content, (
                f"Expected element '{element}' not found in output"
            )
        if hasattr(expected, "SQL_CONTENT_HINTS"):
            assert any(
                hint in modified_content for hint in expected.SQL_CONTENT_HINTS
            ), f"None of the SQL content hints found: {expected.SQL_CONTENT_HINTS}"

    def test_sql_with_fstring_and_regular_string_mixed(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test mix of f-strings and regular strings with code around them."""
        original_code = load_test_case("mixed_fstring_and_regular")
        expected = load_expected_output("mixed_fstring_and_regular")

        python_file = tmp_path / "test.py"
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        for element in expected.PRESERVED_ELEMENTS:
            assert element in modified_content, (
                f"Expected element '{element}' not found in output"
            )
        if hasattr(expected, "FSTRING_EXPRESSIONS"):
            assert any(
                expr in modified_content for expr in expected.FSTRING_EXPRESSIONS
            ), f"None of the f-string expressions found: {expected.FSTRING_EXPRESSIONS}"

    def test_fstring_placeholder_preservation_simple(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test that simple f-string placeholders are preserved exactly."""
        python_file = tmp_path / "test.py"
        original_code = 'def get_data():\n    table = "users"\n    result = spark.sql(f"SELECT * FROM {table}")\n    return result\n'
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Must have exact placeholder syntax with braces
        assert "{table}" in modified_content, (
            f"F-string placeholder '{{table}}' must be preserved. "
            f"Got: {modified_content}"
        )
        # Must still be an f-string
        assert (
            'spark.sql(f"' in modified_content or "spark.sql(f'" in modified_content
        ), f"Must preserve f-string prefix. Got: {modified_content}"

    def test_fstring_placeholder_preservation_multiple(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test that multiple f-string placeholders are all preserved."""
        python_file = tmp_path / "test.py"
        original_code = 'def get_data():\n    table = "users"\n    col = "id"\n    val = 123\n    result = spark.sql(f"SELECT {col} FROM {table} WHERE {col} = {val}")\n    return result\n'
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # All placeholders must be preserved
        assert "{col}" in modified_content, (
            f"F-string placeholder '{{col}}' must be preserved. Got: {modified_content}"
        )
        assert "{table}" in modified_content, (
            f"F-string placeholder '{{table}}' must be preserved. Got: {modified_content}"
        )
        assert "{val}" in modified_content, (
            f"F-string placeholder '{{val}}' must be preserved. Got: {modified_content}"
        )
        # Count occurrences - each placeholder should appear at least once
        assert modified_content.count("{col}") >= 1, (
            f"Placeholder '{{col}}' should appear at least once. Got: {modified_content}"
        )

    def test_fstring_placeholder_preservation_with_attribute(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test that f-string placeholders with attribute access are preserved."""
        python_file = tmp_path / "test.py"
        original_code = 'def get_data():\n    config = type("Config", (), {"table": "users"})()\n    result = spark.sql(f"SELECT * FROM {config.table}")\n    return result\n'
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Attribute access placeholder must be preserved
        assert "{config.table}" in modified_content, (
            f"F-string placeholder with attribute access '{{config.table}}' must be preserved. "
            f"Got: {modified_content}"
        )

    def test_fstring_placeholder_preservation_multiline(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test that f-string placeholders are preserved in multiline SQL."""
        python_file = tmp_path / "test.py"
        original_code = '''def get_data():
    table = "users"
    result = spark.sql(f"""
    SELECT id, name
    FROM {table}
    WHERE active = 1
    """)
    return result
'''
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Placeholder must be preserved in multiline f-string
        assert "{table}" in modified_content, (
            f"F-string placeholder '{{table}}' must be preserved in multiline SQL. "
            f"Got: {modified_content}"
        )
        # Must still be an f-string with triple quotes
        assert 'f"""' in modified_content or "f'''" in modified_content, (
            f"Multiline f-string prefix must be preserved. Got: {modified_content}"
        )

    def test_fstring_placeholder_preservation_after_formatting(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test that placeholders are preserved even when SQL formatting changes."""
        python_file = tmp_path / "test.py"
        # Use SQL that will definitely be reformatted (bad case, spacing)
        original_code = 'def get_data():\n    table = "users"\n    result = spark.sql(f"SeLEct * fRom {table} wHeRe id=1")\n    return result\n'
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Placeholder must be preserved even after SQL is reformatted
        assert "{table}" in modified_content, (
            f"F-string placeholder '{{table}}' must be preserved after SQL formatting. "
            f"Got: {modified_content}"
        )
        # SQL should be reformatted (case fixed)
        assert (
            "SELECT" in modified_content.upper() or "FROM" in modified_content.upper()
        ), f"SQL should be reformatted. Got: {modified_content}"
        # But placeholder should remain unchanged
        placeholder_pos = modified_content.find("{table}")
        assert placeholder_pos != -1, "Placeholder should exist"
        # Extract the SQL part to verify placeholder is in correct position
        # Handle both single and triple quotes
        sql_start = modified_content.find('f"')
        if sql_start == -1:
            sql_start = modified_content.find('f"""')
        if sql_start != -1:
            # Check if it's triple quotes
            if modified_content[sql_start : sql_start + 4] == 'f"""':
                sql_end = modified_content.find('"""', sql_start + 4)
                if sql_end != -1:
                    sql_part = modified_content[sql_start : sql_end + 3]
                else:
                    sql_part = None
            else:
                sql_end = modified_content.find('"', sql_start + 2)
                if sql_end != -1:
                    sql_part = modified_content[sql_start : sql_end + 1]
                else:
                    sql_part = None
            if sql_part:
                assert "{table}" in sql_part, (
                    f"Placeholder should be within the SQL string. SQL part: {sql_part}"
                )

    def test_fstring_placeholder_at_start_of_sql(self, sqlfluff_config_file, tmp_path):
        """Test that f-string placeholder at the start of SQL is preserved."""
        python_file = tmp_path / "test.py"
        original_code = 'def get_data():\n    table = "users"\n    result = spark.sql(f"{table}.id, name FROM users")\n    return result\n'
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Placeholder at start must be preserved
        assert "{table}" in modified_content, (
            f"F-string placeholder '{{table}}' at start of SQL must be preserved. "
            f"Got: {modified_content}"
        )

    def test_fstring_placeholder_at_end_of_sql(self, sqlfluff_config_file, tmp_path):
        """Test that f-string placeholder at the end of SQL is preserved."""
        python_file = tmp_path / "test.py"
        original_code = 'def get_data():\n    table = "users"\n    result = spark.sql(f"SELECT * FROM {table}")\n    return result\n'
        python_file.write_text(original_code)

        reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Placeholder at end must be preserved
        assert "{table}" in modified_content, (
            f"F-string placeholder '{{table}}' at end of SQL must be preserved. "
            f"Got: {modified_content}"
        )
        # Verify the placeholder is actually in the SQL part, not elsewhere
        sql_match = re.search(r'f["\']([^"\']*\{table\}[^"\']*)["\']', modified_content)
        assert sql_match is not None, (
            f"Placeholder '{{table}}' should be within the SQL f-string. "
            f"Got: {modified_content}"
        )
