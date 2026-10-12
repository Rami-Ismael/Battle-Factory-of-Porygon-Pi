"""Conservative, overlapping VGC mode cues; these are not quality assessments.

Web terminology checked 2026-10-11. Mega weather mappings follow the pinned
M-B trace's bundled dex. Keep evidence dependencies for partial-state display.
"""
SPEED_URL = 'https://www.vgcguide.com/speed-control'
WEATHER_URL = 'https://www.trickroomvgc.com/archetypes/'
RULES = {
    'Rain': ({'Drizzle'}, {'Rain Dance'}),
    'Sun': ({'Drought'}, {'Sunny Day'}),
    'Sand': ({'Sand Stream', 'Sand Spit'}, {'Sandstorm'}),
    'Snow': ({'Snow Warning'}, {'Snowscape', 'Chilly Reception'}),
    'Trick Room': (set(), {'Trick Room'}),
    'Tailwind': (set(), {'Tailwind'}),
}
MEGAS = {
    ('charizard', 'Charizardite Y'): ('Sun', 'charizardmegay'),
    ('tyranitar', 'Tyranitarite'): ('Sand', 'tyranitarmega'),
    ('froslass', 'Froslassite'): ('Snow', 'froslassmega'),
    ('abomasnow', 'Abomasite'): ('Snow', 'abomasnowmega'),
}

def classify(record):
    found = {name: [] for name in RULES}
    f = record['fields']
    def add(mode, columns, text):
        found[mode].append({'columns': columns, 'text': text})
    for slot, sp in enumerate(record['speciesId']):
        c = slot*8; name = f[c]
        for mode, (abilities, moves) in RULES.items():
            if f[c+1] in abilities:
                add(mode, [c,c+1], f'{name}: {f[c+1]}')
            for j in range(3,7):
                if f[c+j] in moves:
                    add(mode, [c,c+j], f'{name}: {f[c+j]}')
        for (base, stone), (mode, mega) in MEGAS.items():
            if sp == mega:
                add(mode, [c], f'{name}: weather-setting Mega form (pinned M-B dex)')
            elif sp == base and f[c+2] == stone:
                add(mode, [c,c+2], f'{name}: {stone}, weather access after Mega Evolution')
    return [{'name': mode, 'url': SPEED_URL if mode in ('Trick Room','Tailwind') else WEATHER_URL+mode.lower(),
             'evidence': evidence} for mode,evidence in found.items() if evidence]

def title(modes):
    return ' + '.join(m['name'] for m in modes) if modes else 'Unclassified'
