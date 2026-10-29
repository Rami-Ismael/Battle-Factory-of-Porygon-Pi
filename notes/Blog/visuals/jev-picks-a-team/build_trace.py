"""Write whole-pokemon-trace.json: one hidden Pokémon filled through the real Jev prompt builder.

    /tmp/vgc-pilot/.venv/bin/python build_trace.py

The mask is the meta grid's first whole-Pokémon mask that hides one slot. A scripted chooser
stands in for Jev, which has not been called yet: it takes the first name from PREFER that is on
offer, else the first option. The picks are therefore illustrative. Everything else is what the
real prompt builder produced for this mask: the order of the questions, the number of options in
each, the species tournament's group sizes, and each question's token estimate.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / 'code/scripts'))
import repo_paths  # noqa: E402,F401
import jev_fields  # noqa: E402,F401
import jev_species  # noqa: E402,F401
import jev_fill as J  # noqa: E402

PREFER = ['Garchomp', 'Dragonite', 'Rough Skin', 'Life Orb', 'Choice Scarf', 'Earthquake', 'Dragon Claw', 'Rock Slide',
          'Protect', 'Jolly']
FAST_PHYSICAL = {'Atk': 32, 'Spe': 32}


class Scripted:
    def __init__(self):
        self.asked = []

    def choose(self, state, questions):
        answers = {}
        for name, question in questions.items():
            options = list(question['criteria'])
            spreads = [o for o, d in question['criteria'].items() if isinstance(d, dict) and
                       all(d.get(k) == v for k, v in FAST_PHYSICAL.items())]
            pick = next((p for p in PREFER if p in options), spreads[0] if spreads else options[0])
            self.asked.append({'path': name, 'options': options, 'instructions': question['instructions']})
            answers[name] = {'choice': pick, 'confidence': 1.0, 'probabilities': {o: float(o == pick) for o in options}}
        return answers


def main():
    task = next(t for t in J.tasks('meta') if t['task'] == 'pokemon' and t['k'] == 1)
    chooser = Scripted()
    team, records = J.fill_mask(task, chooser, J.meta_pastes())
    hidden = J.parse_path(task['masked_paths'][0])[0]
    questions = []
    for record, asked in zip(records, chooser.asked, strict=True):
        questions.append({'path': record['path'], 'field': J.parse_path(record['path'])[1:],
                          'round': record.get('round'), 'options': record['options'], 'tokens': record['tokens'],
                          'pick': record['value'], 'sample_options': asked['options'][:6],
                          'instructions': asked['instructions'],
                          'groups': [len(g) for g in record['groups']] if record.get('groups') else None})
    out = {'grid': 'meta', 'task_id': task['id'], 'source': task['source'], 'hidden_slot': hidden,
           'team': [{'species': m['species'], 'item': m['item']} for m in team],
           'filled': team[hidden], 'questions': questions,
           'note': 'picks are scripted (Jev not called yet); option counts, order and token estimates are real'}
    (HERE / 'whole-pokemon-trace.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    for q in questions:
        print(q['path'], q['round'] or '', q['options'], q['tokens'], q['pick'])


if __name__ == '__main__':
    main()
