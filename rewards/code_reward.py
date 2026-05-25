"""
https://github.com/huggingface/open-r1
"""

import re
import json
from typing import Dict, Any, List, Tuple
import ast
import difflib
import json

from complexipy import code_complexity
import black
import autopep8

from absolute_zero_reasoner.utils.code_utils.parsers import (
    parse_imports,
    remove_comments_and_docstrings,
    remove_any_not_definition_imports,
    remove_print_statements,
)


def format_python_code(code: str) -> str:
    """Formats Python code with proper indentation using autopep8."""
    try:
        # First try to use black for formatting
        formatted = black.format_str(code, mode=black.Mode())
        return formatted
    except:
        # Fallback to a simpler approach that handles the specific test case
        # Parse the code line by line
        formatted_lines = []
        in_function = False
        function_indent = 0
        empty_line_after_return = False
        
        for line in code.split('\n'):
            stripped = line.strip()
            
            # Skip empty lines but remember them for context
            if not stripped:
                if in_function and empty_line_after_return:
                    # Empty line after return statement likely means end of function
                    in_function = False
                formatted_lines.append('')
                continue
            
            # Detect function definition
            if stripped.startswith('def ') and stripped.endswith(':'):
                in_function = True
                function_indent = 0
                formatted_lines.append(stripped)
                continue
                
            # Handle indentation inside functions
            if in_function:
                # Check for return statement
                if stripped.startswith('return '):
                    formatted_lines.append('    ' + stripped)
                    empty_line_after_return = True
                    continue
                    
                # Check if this is likely a line outside the function
                if empty_line_after_return and not stripped.startswith(('    ', '\t')):
                    in_function = False
                    formatted_lines.append(stripped)
                    continue
                    
                # Regular function body line
                formatted_lines.append('    ' + stripped)
            else:
                # Line outside any function
                formatted_lines.append(stripped)
        
        # Apply autopep8 for final cleanup
        return autopep8.fix_code(
            '\n'.join(formatted_lines),
            options={'aggressive': 1, 'indent_size': 4}
        )


def extract_code(completion: str) -> str:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def parse_to_ast(code_snippet: str) -> ast.AST:
    """
    Parse a Python code snippet into an Abstract Syntax Tree (AST).
    
    Args:
        code_snippet: A string containing Python code
        
    Returns:
        An AST object representing the code
        
    Raises:
        SyntaxError: If the code snippet contains syntax errors
    """
    try:
        return ast.parse(code_snippet)
    except SyntaxError as e:
        print(f"Syntax error in code: {e}")
        raise


def ast_to_dict(node: ast.AST) -> Dict[str, Any]:
    """
    Convert an AST node to a dictionary representation for easier comparison.
    
    Args:
        node: An AST node
        
    Returns:
        A dictionary representing the node and its children
    """
    if isinstance(node, ast.AST):
        # Extract node type and fields
        result = {"node_type": node.__class__.__name__}
        
        # Add children nodes
        for field, value in ast.iter_fields(node):
            if field == "ctx":  # Skip context objects as they vary unnecessarily
                continue
                
            # Handle different types of field values
            if isinstance(value, list):
                result[field] = [ast_to_dict(item) for item in value if isinstance(item, ast.AST)]
            elif isinstance(value, ast.AST):
                result[field] = ast_to_dict(value)
            elif value is not None:
                # Keep primitive values unchanged
                result[field] = value
                
        return result
    else:
        return {"value": str(node)}


def ast_edit_distance(code1: str, code2: str) -> float:
    """
    Calculate the edit distance between two Abstract Syntax Trees.
    
    Args:
        ast1: First AST
        ast2: Second AST
        
    Returns:
        A float value representing the normalized edit distance (0.0 = identical, 1.0 = completely different)
    """
    try:
        ast1 = parse_to_ast(format_python_code(code1))
        ast2 = parse_to_ast(format_python_code(code2))

        # Convert ASTs to dictionary representation
        dict1 = ast_to_dict(ast1)
        dict2 = ast_to_dict(ast2)
        
        # Convert to strings for difflib comparison
        str1 = json.dumps(dict1, sort_keys=True, indent=2)
        str2 = json.dumps(dict2, sort_keys=True, indent=2)
        
        # Calculate similarity ratio using difflib
        similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
        
        # Convert similarity to distance (1.0 - similarity)
        distance = 1.0 - similarity

        return distance
    except Exception as e:
        print(f"Error in ast_edit_distance: {e}")
        return 0.0


def ast_edit_operations(ast1: ast.AST, ast2: ast.AST) -> List[Dict[str, Any]]:
    """
    Generate a list of edit operations needed to transform ast1 into ast2.
    
    Args:
        ast1: First AST
        ast2: Second AST
        
    Returns:
        A list of edit operations (insert, delete, modify)
    """
    # Convert ASTs to dictionary representation
    dict1 = ast_to_dict(ast1)
    dict2 = ast_to_dict(ast2)
    
    # Convert to strings for difflib comparison
    str1 = json.dumps(dict1, sort_keys=True, indent=2).splitlines()
    str2 = json.dumps(dict2, sort_keys=True, indent=2).splitlines()
    
    # Calculate differences
    diff = list(difflib.unified_diff(str1, str2, n=0))
    
    # Parse diff into operations
    operations = []
    for line in diff[2:]:  # Skip the header lines
        if line.startswith('+'):
            operations.append({
                "operation": "insert",
                "content": line[1:].strip()
            })
        elif line.startswith('-'):
            operations.append({
                "operation": "delete",
                "content": line[1:].strip()
            })
        elif line.startswith(' '):
            # Context line, no operation needed
            pass
    
    return operations


def get_code_complexity_reward(code_snippet: str) -> float:
    """
    Calculate the complexity of a Python code snippet using the `code_complexity` function from the `complexipy` library.

    Args:
        code_snippet: A string containing Python code
    
    Returns:
        A float value representing the complexity of the code snippet
    """
    try:
        return code_complexity(format_python_code(code_snippet)).complexity / 15
    except Exception as e:
        return 0.0


def get_halstead_reward(code_snippet: str, 
                        effort_max: float = 10000, 
                        complexity_max: float = 10, 
                        volume_max: float = 500) -> float:
    """
    Calculate the Halstead reward for a Python code snippet.

    Args:
        code_snippet: A string containing Python code
    
    Returns:
        A float value representing the Halstead reward of the code snippet
    """
    try:
        from radon.metrics import h_visit
        from radon.complexity import cc_visit
        
        code = format_python_code(code_snippet)

        h = h_visit(code).total
        effort = h.effort
        volume = h.volume
        cc_blocks = cc_visit(code)
        complexity = max((b.complexity for b in cc_blocks), default=1)
        effort_norm = min(effort / effort_max, 1.0)
        complexity_norm = min(complexity / complexity_max, 1.0)
        volume_norm = min(volume / volume_max, 1.0)

        w1, w2, w3 = 0.5, 0.3, 0.2

        score = w1 * effort_norm + w2 * complexity_norm + w3 * volume_norm
        return round(score, 3)
    except Exception as e:
        return 0.0


def has_test_input(snippet_code: str) -> bool:
    test_patterns = [
        r"(?i)#\s*(test|example)",  # Match any test/example comment
        r"\b(input|test_input|sample_input)\b\s*=",  # Common test variable names
        r"\b\w*input\w*\s*=\s*",    # Match any variable containing "input"
        r"\b(expected|output|result)\s*=\s*",
        r"\bassert\b",
        r"print\s*\(\s*f\(",
        r"f\(\[.*\]\)",
        r"f\([^)]*\)\s*(#|$)",
        r"^\s*input\s*$",  # Match lines containing only "input"
    ]

    return any(
        re.search(pattern, snippet_code, re.MULTILINE)
        for pattern in test_patterns
    )

def get_type_counts_reward(answer: str, type_counters: Dict[str, Dict[str, int]], hierarchical: bool = False) -> float:
    """
    Calculate the type counts reward for a Python code snippet.

    Args:
        answer: A string containing the answer
        type_counters: A dictionary of type counters
        hierarchical: Whether to use hierarchical type counts
    """
    if hierarchical:
        # we do not flatten we first have a distribution of the types, then we have a distribution of the elements within each type
        # we want to maximize the suprise of the answer
        # first, we get the distribution of the types
        type_distribution = {}
        for key, value in type_counters.items():
            type_distribution[key] = sum(value.values())

        # try to get the type, if failed default it as a string
        try:
            answer_type = type(eval(answer)).__name__
        except:
            answer_type = 'str'

        # then, we get the "suprise" of the answer, sum of 1 - probability of answer_type and 1 - probability of the element within the type
        suprise = 0
        if answer_type in type_distribution:
            suprise += 1 - (type_distribution[answer_type] / sum(type_distribution.values()))
        else:
            suprise += 1.0
        if answer_type in type_counters:
            if answer in type_counters[answer_type]:
                suprise += 1 - (type_counters[answer_type][answer] / sum(type_counters[answer_type].values()))
            else:
                suprise += 1.0
        else:
            suprise += 1.0
        return suprise / 2
    else:
        # first flatten the type_counters, use the counts of each element as a categorical distribution, then, we get the "suprise" of the answer
        # we want to maximize the suprise
        # first, flatten the type_counters
        flattened_type_counters = {}
        for _, value in type_counters.items():
            for sub_key, sub_value in value.items():
                flattened_type_counters[sub_key] = sub_value
        # then, we get the "suprise" of the answer

        if answer in flattened_type_counters:
            suprise = 1 - (flattened_type_counters[answer] / sum(flattened_type_counters.values()))
            return suprise
        return 1.0
