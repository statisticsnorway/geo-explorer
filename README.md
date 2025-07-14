# geo-explorer

Explore geodata interactively.

Opprettet av:
ort <ort@ssb.no>

---

## GeoExplorer


```python
from geo_explorer import GeoExplorer
from geo_explorer import LocalFileSystem
explorer = GeoExplorer(
    start_dir="C:/users/ort/OneDrive - Statistisk sentralbyrå/data",
    file_system=LocalFileSystem(),
    port=8055,
)
explorer.run(debug=True)
```

Run locally:

```python
from geo_explorer import GeoExplorer
from geo_explorer import LocalFileSystem
explorer = GeoExplorer(
    start_dir="C:/users/user/data",
    file_system=LocalFileSystem(),
    port=None,
)
explorer.run()
```
