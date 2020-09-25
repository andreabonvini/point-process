# PPLIB - The Point Process Library

This repository contains a `Python` implementation of the `MATLAB` software provided by Riccardo Barbieri and Luca Citi at [this](http://users.neurostat.mit.edu/barbieri/pphrv) link.

## Development environment:

The development environment follows some best practices to keep code working and clean. In particular, before commit and pushing the project uses:
- [black](https://github.com/psf/black): reformats code using conventions
- [isort](https://github.com/timothycrosley/isort): sorts imports automatically
- [flake8](https://github.com/PyCQA/flake8): _lints_ (=runs quality tools) the code
- [pytest](https://github.com/pytest-dev/pytest): tests the code with coverage


### Setup:
```bash
# Run these commands the first time you clone this repository

# Install dependencies
pipenv install --dev

# Setup pre-commit and pre-push hooks
pipenv run pre-commit install -t pre-commit
pipenv run pre-commit install -t pre-push

# -----------------------------------------
# Run these commands when needed:

# Install new packages
pipenv install <package-name>

# Update the dependencies
pipenv sync

# Run bin/*.py script
pipenv run python bin/*.py
```

### Tests

Only the functions which are contained in the `pp` module are tested with full coverage.

```bash
# Run tests
pipenv run pytest
# Run tests with coverage report
pipenv run pytest --cov --cov-report term-missing
# Run and profile tests
pipenv run pytest --duration=0
```
