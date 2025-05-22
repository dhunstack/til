# Setting up TIL blog

I took the template from [simonw/til](https://github.com/remy/simonw-til-template).

Had to add the following permissions in `.github/workflows/build.yml` -

```yaml
on:
  push:
    branches:
      - main    # instead of master

permissions:
    contents: write  # This gives permission to push changes
```

Fixed the `build_database.py` file -

```python
def created_changed_times(repo_path, ref="main"):  # Change "master" to "main" here
```

Also had to update Github Actions configuration -

```yaml
- name: Set up Python
uses: actions/setup-python@v4  # Updated to v4
with:
    python-version: '3.12'  # Updated to a newer Python version
- uses: actions/cache@v3  # Updated to v3
```

To test the workflow locally with docker, I used 

```bash
# macOS with Homebrew
brew install act
act -j build
```

To conduct manual test locally, can also use

```bash
python build_database.py
python update_readme.py --rewrite
```