# Contributing to OpenGNC

First off, thank you for considering contributing to the OpenGNC! It's people like you who make this a great tool for the spacecraft engineering community.

## Code of Conduct

Please be respectful and professional in all interactions.

## How Can I Contribute?

### Reporting Bugs
- Use the **GitHub Issues** tab to report bugs.
- Include a reproducible example and the expected vs actual behavior.

### Suggesting Enhancements
- Open an Issue with the tag `enhancement`.
- Describe the feature, why it is useful, and potential implementation details.

### Pull Requests
1. Fork the repository and create your branch from `main`.
2. Ensure your code follows the style guides below.
3. Add unit tests for any new functionality.
4. Verify all tests pass with `pytest`.
5. Submit a Pull Request.

## Style Guides

### Python Style Guide
- We follow **PEP 8** guidelines.
- Use **NumPy Style** docstrings for all public APIs.
  ```python
  def function(arg1, arg2):
      \"\"\"
      Summary line.

      Parameters
      ----------
      arg1 : type
          Description of arg1.
      arg2 : type
          Description of arg2.

      Returns
      -------
      type
          Description of return value.
      \"\"\"
      pass
  ```

### Testing
- Run tests using `pytest`:
  ```bash
  pytest
  ```
- We aim for high test coverage. You can check coverage with:
  ```bash
  pytest --cov=src
  ```

## Development Setup

1. Clone your fork.
2. Install in editable mode with dev dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Run `pytest` to ensure everything works.

---
*Thank you for your contributions!*




