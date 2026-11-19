# The environment the results were produced with: Python 3.14.7 and the exact package versions of
# requirements-lock.txt. Not yet built by the team (the laptop has no Docker); the lock file was
# made on Windows, and every package in it also installs on Linux.
#
#   docker build -t pace-cfmnn .
#   docker run --rm -v "$PWD/data:/app/data" pace-cfmnn python -B scripts/download_data.py
#   docker run --rm -v "$PWD/data:/app/data" pace-cfmnn            # the full test suite
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" pace-cfmnn \
#       python -B scripts/<run script>.py ...                      # one table (README)
#
# The datasets are mounted, not copied: scripts/download_data.py fetches them and checks each
# file's SHA-256 against the copy the runs used.
FROM python:3.14.7-slim

WORKDIR /app
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY . .
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e . || pip install --no-cache-dir --no-deps -e .

CMD ["python", "-B", "-m", "unittest", "discover", "-s", "tests", "-t", "."]
