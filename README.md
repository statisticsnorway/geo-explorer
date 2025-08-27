# geo-explorer

Explore geodata interactively with a file browser.

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

Create a simple python file:

```python
from geo_explorer import GeoExplorer
from gcsfs import GCSFileSystem

DELT_KART = "ssb-areal-data-delt-kart-prod"
YEAR = 2025

GeoExplorer(
    start_dir=f"{DELT_KART}/analyse_data/klargjorte-data/{YEAR}",
    favorites=[
        f"{DELT_KART}/analyse_data/klargjorte-data/{YEAR}",
        f"{DELT_KART}/visualisering_data/klargjorte-data/{YEAR}/parquet",
    ],
    center=(59.91740845, 10.71394444),
    zoom=13,
    file_system=GCSFileSystem(),
    port=3000,
).run()
```

And run it:

```shell
poetry run python my_file.py
# or
python my_file.py
```

### Export as code
The export button can be used to "save" your map. Copy the printed code and paste it into a new file. Running this file will give you an app with the same bounds, data and coloring.

### Starting the app with data loaded, filtered and colored
Here is an example of a GeoExplorer app where data is loaded, filtered and colored:

```python
import sgis as sg

entur_path = f"{DELT_KART}/analyse_data/klargjorte-data/{YEAR}/ENTUR_Holdeplasser_punkt_p{YEAR}_v1.parquet"

# Create a custom GeoDataFrame to add to the map
jernbanetorget = sg.to_gdf([10.7535581, 59.9110967], crs=4326).to_crs(25833)
jernbanetorget.geometry = jernbanetorget.buffer(500)

GeoExplorer(
    start_dir=f"{DELT_KART}/analyse_data/klargjorte-data/{YEAR}",
    favorites=[
        f"{DELT_KART}/analyse_data/klargjorte-data/{YEAR}",
        f"{DELT_KART}/visualisering_data/klargjorte-data/{YEAR}/parquet",
    ],
    data={
        "jernbanetorget_500m": jernbanetorget,
        entur_path: "kjoeretoey != 'fly'",
    },
    column="kjoeretoey",
    color_dict={
        "jernbane": "darkgreen",
        "buss": "red",
        "trikk": "deepskyblue",
        "tbane": "yellow",
        "baat": "navy",
    },
    center=(59.91740845, 10.71394444),
    zoom=13,
    file_system=GCSFileSystem(),
    port=3000,
).run()
```

The 'data' argument can be either:
- a list of file paths
- a dict with labels as keys and GeoDataFrames as values
- a dict with file paths as keys and filter function (or None) as value (note that the filter function must be formated as a string!)

Set the 'column' argument to color the geometries.

Optionally specify what colors with the 'color_dict' argument (with column values as dict keys and color codes (hex) or named colors (https://matplotlib.org/stable/gallery/color/named_colors.html) as dict values).

### Filter functions
Filtering data can be done in the GeoExplorer init, as shown above, or in the app.

Filter functions can be:
- polars expression or an iterable of such, e.g.: *(pl.col("FYLKE") != "50", pl.col("FYLKE") != "03")*
- lambda functions accepted by pandas.loc, e.g. *lambda x: x["kjoeretoey"] != "fly"*
- queries accepted by pandas.query, e.g. *kjoeretoey != "fly"*

Note that the filter functions must be wrapped in quotation marks ("") if used in the GeoExplorer init.

For large datasets, the polars approach might be noticeably faster, both because polars is faster and because the data is stored as polars.DataFrames and converted to pandas and back if the polars filtering fails.

### Local files
For local files, use the LocalFileSystem class, which simply implements glob and ls methods based on the standard library (os and glob).

```python
from geo_explorer import GeoExplorer
from geo_explorer import LocalFileSystem

GeoExplorer(
    start_dir="ssb-areal-data-delt-kart-prod/analyse_data/klargjorte-data/2025",
    file_system=GCSFileSystem(),
).run()
```

Other file systems can be used, as long as it acts like fsspec's AbstractFileSystem and implement the methods *ls* and *glob*. The methods should take the argument 'detail', which, if set to True, will return a dict for each listed path with the keys "updated" (timestamp), "size" (bytes), "name" (full path) and "type" ("directory" or "file").

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
