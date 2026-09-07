# harvard_chan

Coursework repo for BST 220.

## Setup

This repo uses a Python 3.10 virtual environment, kept *outside* the repo
(as a sibling folder, e.g. `../.venv-3.10`) so it never gets committed.

```
# create the venv (one time)
python3.10 -m venv ../.venv-3.10

# activate it (every new terminal session)
source ../.venv-3.10/bin/activate

# install all required packages
pip install -r requirements.txt
```

## Keeping requirements.txt up to date

Whenever you install a new package for an assignment:

```
pip install <package-name>
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Add <package-name> dependency"
```

This keeps `requirements.txt` pinned to exact versions, so re-running
`pip install -r requirements.txt` always reproduces the same environment.
