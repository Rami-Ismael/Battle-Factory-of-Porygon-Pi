`GET https://replay.pokemonshowdown.com/<id>.json` returns:

```json
{"id": "gen9championsvgc2026regmb-2669445930",
 "format": "[Gen 9 Champions] VGC 2026 Reg M-B",
 "formatid": "gen9championsvgc2026regmb",
 "players": ["regitron 21", "gg cyovlu"],
 "uploadtime": 1787521474, "views": "1", "rating": 1032,
 "private": 0, "password": null,
 "log": "|j|☆regitron 21\n|t:|1787521154\n|gametype|doubles\n..."}
```

`scrape_logs.py` keeps only `id`, `uploadtime`, `log` — the rest, including `rating`, is thrown away and later re-parsed out of the log text by `get_rating`.

# You can tell the log is very off looking
- You can split the log using the regex \n|\n"