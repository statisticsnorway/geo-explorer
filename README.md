# geo-explorer

Explore geodata interactively in an app.

Opprettet av:
ort <ort@ssb.no>

---

To install, use either:

```shell
poetry add geo-explorer
```

Or:

```shell
pip install geo-explorer
```

## GeoExplorer

It's best to run the app in the terminal.

Create a python file like this:

```python
from geo_explorer import GeoExplorer
from geo_explorer import LocalFileSystem

explorer = GeoExplorer(
    start_dir="/buckets/delt-kart/analyse_data/klargjorte-data/2025",
    favorites=[
        "/buckets/delt-kart/analyse_data/klargjorte-data/2025",
        "/buckets/delt-kart/visualisering_data/klargjorte-data/2025/parquet",
    ],
    zoom=13,
    center=(59.91740845, 10.71394444),
    file_system=LocalFileSystem(),
    port=3000,
).run()
```

And run the file.

You can also use other file systems, for instance GCSFileSystem for Google Cloud Storage:

```python
from geo_explorer import GeoExplorer
from gcsfs import GCSFileSystem

GeoExplorer(
    start_dir="ssb-areal-data-delt-kart-prod/analyse_data/klargjorte-data/2025",
    favorites=[
        "ssb-areal-data-delt-kart-prod/analyse_data/klargjorte-data/2025",
        "ssb-areal-data-delt-kart-prod/visualisering_data/klargjorte-data/2025/parquet",
    ],
    zoom=13,
    center=(59.91740845, 10.71394444),
    file_system=GCSFileSystem(),
    port=3000,
).run()
```

The file system should act like fsspec's AbstractFileSystem and implement the methods *ls* and *glob*.

## Developer information

### Git LFS

The data in the testdata directory is stored with [Git LFS](https://git-lfs.com/).
Make sure `git-lfs` is installed and that you have run the command `git lfs install`
at least once. You only need to run this once per user account.

### Dependencies

[Poetry](https://python-poetry.org/) is used for dependency management. Install
poetry and run the command below from the root directory to install the dependencies.

```shell
poetry install -E test --no-root
```

### Tests

Use the following command from the root directory to run the tests:

```shell
poetry run pytest  # from root directory
```

For VS Code there are extensions for opening a python script as Jupyter Notebook,
for example:
[Jupytext for Notebooks](https://marketplace.visualstudio.com/items?itemName=donjayamanne.vscode-jupytext).

### Code quality

Run 'ruff' on all files with safe fixes:

```shell
poetry run ruff check --fix .
```

### Formatting

Format the code with `black` and `isort` by running the following command from the
root directory:

```shell
poetry run black .
poetry run isort .
```

### Pre-commit hooks

We are using [pre-commit hooks](https://pre-commit.com/) to make sure the code is
correctly formatted and consistent before committing. Use the following command from
the root directory in the repo to install the pre-commit hooks:

```shell
poetry run pre-commit install
```

It then checks the changed files before committing. You can run the pre-commit checks
on all files by using this command:

```shell
poetry run pre-commit run --all-files
```

### Documentation

To generate the API-documentation locally, run the following command from the root
directory:

```shell
poetry run sphinx-build -W docs docs/_build
```

Then open the file `docs/_build/index.html`.

To check and run the docstrings examples, run this command:

```shell
poetry run xdoctest --command=all ./src/sgis
```

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_SSB sgis_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [Statistics Norway]'s [SSB PyPI Template].

[statistics norway]: https://www.ssb.no/en
[pypi]: https://pypi.org/
[ssb pypi template]: https://github.com/statisticsnorway/ssb-pypitemplate
[file an issue]: https://github.com/statisticsnorway/ssb-sgis/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/statisticsnorway/ssb-sgis/blob/main/LICENSE
[contributor guide]: https://github.com/statisticsnorway/ssb-sgis/blob/main/CONTRIBUTING.md
[reference guide]: https://statisticsnorway.github.io/ssb-sgis/reference.html
