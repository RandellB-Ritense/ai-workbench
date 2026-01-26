"""
I/O utilities for reading and writing data in various formats.
"""
import json
from pathlib import Path
from typing import Any, Dict, List


def save_json(data: Any, output_path: Path, indent: int = 2):
    """
    Save data to a JSON file.

    Args:
        data: Data to save (must be JSON serializable)
        output_path: Path to output file
        indent: JSON indentation (default: 2)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(input_path: Path) -> Any:
    """
    Load data from a JSON file.

    Args:
        input_path: Path to input file

    Returns:
        Loaded JSON data
    """
    with open(input_path, 'r') as f:
        return json.load(f)


def save_text(content: str, output_path: Path):
    """
    Save text content to a file.

    Args:
        content: Text content to save
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)


def load_text(input_path: Path) -> str:
    """
    Load text content from a file.

    Args:
        input_path: Path to input file

    Returns:
        Text content
    """
    with open(input_path, 'r') as f:
        return f.read()


def save_lines(lines: List[str], output_path: Path):
    """
    Save list of lines to a file.

    Args:
        lines: List of strings to save
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def load_lines(input_path: Path) -> List[str]:
    """
    Load lines from a file.

    Args:
        input_path: Path to input file

    Returns:
        List of lines
    """
    with open(input_path, 'r') as f:
        return [line.strip() for line in f.readlines()]
