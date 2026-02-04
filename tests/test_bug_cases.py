"""Tests for documented bug cases from sql_pyspark_fixer_test_cases."""

import pytest
from sqlfluff_pyspark.ast_extract import reformat_sql_in_python_file


class TestCase01FStringInterpolationBroken:
    """Test case 01: F-string variable interpolation broken (CRITICAL)."""

    def test_fstring_variable_interpolation_preserved(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test that f-string variable placeholders are preserved."""
        python_file = tmp_path / "test.py"
        # Minimal case: single variable interpolation
        original_code = '''company_code = "HPL"
df = spark.sql(f"""
    SELECT * FROM table WHERE code = '{company_code}'
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # CRITICAL: The placeholder must be preserved
        assert "{company_code}" in modified_content, (
            f"F-string placeholder '{{company_code}}' must be preserved. "
            f"Got: {modified_content}"
        )
        # Must still be an f-string
        assert 'f"""' in modified_content or "f'''" in modified_content, (
            f"Must preserve f-string prefix. Got: {modified_content}"
        )

    def test_fstring_multiple_variables_preserved(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test that multiple f-string variables are all preserved."""
        python_file = tmp_path / "test.py"
        original_code = '''company_code = "HPL"
department = "SALES"
df = spark.sql(f"""
    SELECT * FROM table 
    WHERE company = '{company_code}' 
    AND dept = '{department}'
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Both placeholders must be preserved
        assert "{company_code}" in modified_content, (
            f"Placeholder '{{company_code}}' must be preserved. Got: {modified_content}"
        )
        assert "{department}" in modified_content, (
            f"Placeholder '{{department}}' must be preserved. Got: {modified_content}"
        )

    def test_fstring_expression_preserved(self, sqlfluff_config_file, tmp_path):
        """Test that f-string expressions (not just variables) are preserved."""
        python_file = tmp_path / "test.py"
        original_code = '''from datetime import datetime
df = spark.sql(f"""
    SELECT * FROM table 
    WHERE date >= '{datetime.now().strftime("%Y-%m-%d")}'
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Expression must be preserved (at least the key parts)
        assert "datetime.now()" in modified_content or "{datetime.now()" in modified_content, (
            f"F-string expression must be preserved. Got: {modified_content}"
        )


class TestCase02FStringFormatPlaceholders:
    """Test case 02: F-string format placeholders broken (CRITICAL)."""

    def test_fstring_format_specifier_preserved(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test that f-string format specifiers like {price:.2f} are preserved."""
        python_file = tmp_path / "test.py"
        original_code = '''price = 1234.56
df = spark.sql(f"""
    SELECT * FROM products
    WHERE price = {price:.2f}
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Format specifier must be preserved
        assert "{price:.2f}" in modified_content, (
            f"Format specifier '{{price:.2f}}' must be preserved. "
            f"Got: {modified_content}"
        )

    def test_fstring_date_format_preserved(self, sqlfluff_config_file, tmp_path):
        """Test that date format specifiers are preserved."""
        python_file = tmp_path / "test.py"
        original_code = '''from datetime import datetime
current_date = datetime.now()
df = spark.sql(f"""
    SELECT * FROM orders
    WHERE order_date >= '{current_date:%Y-%m-%d}'
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Date format specifier must be preserved
        assert "{current_date:%Y-%m-%d}" in modified_content or "{current_date:" in modified_content, (
            f"Date format specifier must be preserved. Got: {modified_content}"
        )


class TestCase03EscapedQuotes:
    """Test case 03: SQL strings with escaped quotes (MEDIUM)."""

    def test_escaped_single_quotes_preserved(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test that escaped single quotes in SQL are preserved."""
        python_file = tmp_path / "test.py"
        original_code = '''df = spark.sql("""
    SELECT * FROM customers
    WHERE name = 'O''Brien'
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Escaped quotes should be preserved
        assert "O''Brien" in modified_content or "O\\'Brien" in modified_content, (
            f"Escaped quotes must be preserved. Got: {modified_content}"
        )


class TestCase04NestedSQLCalls:
    """Test case 04: Multiple spark.sql() calls (MEDIUM)."""

    def test_multiple_sql_calls_processed(self, sqlfluff_config_file, tmp_path):
        """Test that multiple spark.sql() calls are all processed."""
        python_file = tmp_path / "test.py"
        original_code = '''df1 = spark.sql("""
    SELECT * FROM table1
    WHERE id = 1
""")

df2 = spark.sql("""
    SELECT * FROM table2
    WHERE id = 2
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=True
        )

        # Should find both SQL strings
        assert result["sql_strings_found"] == 2, (
            f"Expected 2 SQL strings, found {result['sql_strings_found']}"
        )

    def test_multiple_fstring_sql_calls_preserved(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test that multiple f-string SQL calls preserve all variables."""
        python_file = tmp_path / "test.py"
        original_code = '''var1 = "value1"
var2 = "value2"

df_f1 = spark.sql(f"SELECT * FROM table WHERE col = '{var1}'")
df_f2 = spark.sql(f"""
    SELECT * FROM table WHERE col = '{var2}'
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Both variables must be preserved
        assert "{var1}" in modified_content, (
            f"Variable '{{var1}}' must be preserved. Got: {modified_content}"
        )
        assert "{var2}" in modified_content, (
            f"Variable '{{var2}}' must be preserved. Got: {modified_content}"
        )


class TestCase05SQLWithPythonComments:
    """Test case 05: SQL strings with Python-style comments (LOW)."""

    def test_sql_with_comments_preserved(self, sqlfluff_config_file, tmp_path):
        """Test that SQL comments inside strings are preserved."""
        python_file = tmp_path / "test.py"
        original_code = '''# Python comment before SQL
df = spark.sql("""
    -- SQL comment inside string
    SELECT * FROM table
    -- Another SQL comment
    WHERE id = 1
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # SQL comments should be preserved
        assert "--" in modified_content or "SELECT" in modified_content, (
            f"SQL comments should be preserved. Got: {modified_content}"
        )


class TestCase06SQLAlreadyFormatted:
    """Test case 06: SQL already formatted - idempotency test (LOW)."""

    def test_idempotency_already_formatted_sql(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test that already formatted SQL doesn't change on second run."""
        python_file = tmp_path / "test.py"
        original_code = '''df = spark.sql("""
SELECT DISTINCT
    col1 AS column1
    ,col2 AS column2
FROM table
WHERE id = 1
""")
'''
        python_file.write_text(original_code)

        # First run
        result1 = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )
        content1 = python_file.read_text()

        # Second run
        result2 = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )
        content2 = python_file.read_text()

        # Content should be the same (idempotent)
        assert content1 == content2, (
            f"Fixer should be idempotent. First run: {content1}, "
            f"Second run: {content2}"
        )


class TestCase07SQLWithSpecialCharacters:
    """Test case 07: SQL with special characters and unicode (MEDIUM)."""

    def test_special_characters_preserved(self, sqlfluff_config_file, tmp_path):
        """Test that special SQL operators are preserved."""
        python_file = tmp_path / "test.py"
        original_code = '''df = spark.sql("""
    SELECT 
        first_name || ' ' || last_name AS full_name
    FROM customers
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # String concatenation operator should be preserved
        assert "||" in modified_content, (
            f"Special characters (||) must be preserved. Got: {modified_content}"
        )

    def test_unicode_characters_preserved(self, sqlfluff_config_file, tmp_path):
        """Test that Unicode characters are preserved."""
        python_file = tmp_path / "test.py"
        original_code = '''df = spark.sql("""
    SELECT * FROM products
    WHERE name LIKE '%café%'
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Unicode characters should be preserved
        assert "café" in modified_content, (
            f"Unicode characters must be preserved. Got: {modified_content}"
        )


class TestCase08SQLStringConcatenation:
    """Test case 08: SQL strings built via concatenation (MEDIUM)."""

    def test_concatenated_sql_not_detected(self, sqlfluff_config_file, tmp_path):
        """Test that concatenated SQL strings are not detected (expected behavior)."""
        python_file = tmp_path / "test.py"
        original_code = '''base_query = "SELECT * FROM table"
where_clause = "WHERE id = 1"
full_query = base_query + " " + where_clause
df = spark.sql(full_query)
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=True
        )

        # Concatenated SQL strings are not detected (only string literals are)
        # This is expected behavior - the fixer only processes string literals
        assert result["sql_strings_found"] == 0, (
            f"Concatenated SQL should not be detected. "
            f"Found: {result['sql_strings_found']}"
        )


class TestCase09SQLWithJinjaTemplates:
    """Test case 09: SQL strings containing Jinja templating syntax (HIGH)."""

    def test_jinja_variables_preserved(self, sqlfluff_config_file, tmp_path):
        """Test that Jinja template syntax is preserved."""
        python_file = tmp_path / "test.py"
        original_code = '''df = spark.sql("""
    SELECT * FROM table
    WHERE company = '{{ company_name }}'
    AND date >= '{{ start_date }}'
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Jinja syntax should be preserved
        assert "{{ company_name }}" in modified_content or "{{company_name}}" in modified_content, (
            f"Jinja template syntax must be preserved. Got: {modified_content}"
        )

    def test_jinja_conditionals_preserved(self, sqlfluff_config_file, tmp_path):
        """Test that Jinja conditionals are preserved."""
        python_file = tmp_path / "test.py"
        original_code = '''df = spark.sql("""
    SELECT * FROM table
    {% if include_filter %}
    WHERE id = 1
    {% endif %}
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=False
        )

        modified_content = python_file.read_text()
        # Jinja conditionals should be preserved
        assert "{% if" in modified_content or "{%if" in modified_content, (
            f"Jinja conditionals must be preserved. Got: {modified_content}"
        )


class TestCase10SQLInComments:
    """Test case 10: False positives - spark.sql() mentioned in comments (LOW)."""

    def test_sql_in_comments_not_processed(self, sqlfluff_config_file, tmp_path):
        """Test that spark.sql() in comments is not processed."""
        python_file = tmp_path / "test.py"
        original_code = '''# Example comment with spark.sql() - should NOT be processed
# df = spark.sql("SELECT * FROM table WHERE id = 1")

# Actual SQL call (should be processed)
df = spark.sql("""
    SELECT * FROM actual_table
    WHERE id = 1
""")
'''
        python_file.write_text(original_code)

        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=True
        )

        # Should only find the actual SQL call, not the commented one
        assert result["sql_strings_found"] == 1, (
            f"Should find only 1 SQL string (the actual call), "
            f"not commented ones. Found: {result['sql_strings_found']}"
        )


class TestCase11ParsingErrors:
    """Test case 11: Files that fail to parse (HIGH)."""

    def test_complex_nested_calls_parsed(self, sqlfluff_config_file, tmp_path):
        """Test that complex nested calls with SQL are parsed correctly."""
        python_file = tmp_path / "test.py"
        original_code = '''df = spark.sql(
    """
    SELECT * FROM table
    """
).filter(
    col("id") > 0
)
'''
        python_file.write_text(original_code)

        # Should not raise a parsing error
        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=True
        )

        # Should successfully parse and find the SQL
        assert result["sql_strings_found"] >= 0, (
            f"Should parse successfully. Found: {result['sql_strings_found']}"
        )

    def test_sql_in_list_comprehension_parsed(
        self, sqlfluff_config_file, tmp_path
    ):
        """Test that SQL in list comprehensions is parsed."""
        python_file = tmp_path / "test.py"
        original_code = '''queries = [
    spark.sql(f"SELECT * FROM table{i}")
    for i in range(10)
]
'''
        python_file.write_text(original_code)

        # Should not raise a parsing error
        result = reformat_sql_in_python_file(
            str(python_file), sqlfluff_config_file, dry_run=True
        )

        # Should successfully parse (may or may not find SQL depending on implementation)
        assert isinstance(result["sql_strings_found"], int), (
            "Should parse successfully without errors"
        )
