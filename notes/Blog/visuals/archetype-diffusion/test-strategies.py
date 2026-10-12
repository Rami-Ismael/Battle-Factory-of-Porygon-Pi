"""Check positive cues, misleading species matches, hybrids and trace coverage."""
import json
import unittest
from collections import Counter
from pathlib import Path
from strategies import classify, title

def team():
    return {'speciesId':['unknown']*6, 'fields':['Unknown','None','None','None','None','None','None','Serious']*6}

class StrategiesTest(unittest.TestCase):
    def test_hybrid_keeps_every_mode(self):
        r=team();r['fields'][1]='Drizzle';r['fields'][3]='Trick Room';r['fields'][4]='Tailwind'
        self.assertEqual([m['name'] for m in classify(r)],['Rain','Trick Room','Tailwind'])
        self.assertEqual(classify(r)[0]['evidence'][0]['columns'],[0,1])

    def test_species_is_not_ability(self):
        r=team();r['speciesId'][0]='tyranitar';r['fields'][0]='Tyranitar';r['fields'][1]='Unnerve'
        self.assertEqual(classify(r),[])
        self.assertEqual(title(classify(r)),'Unclassified')

    def test_mega_requires_compatible_species(self):
        r=team();r['fields'][2]='Charizardite Y'
        self.assertEqual(classify(r),[])
        r['speciesId'][0]='charizard';r['fields'][0]='Charizard'
        self.assertEqual(classify(r)[0]['name'],'Sun')
        self.assertEqual(classify(r)[0]['evidence'][0]['columns'],[0,2])

    def test_manual_weather_and_activation_abilities(self):
        for cue,mode in [('Rain Dance','Rain'),('Sunny Day','Sun'),('Sandstorm','Sand'),('Snowscape','Snow'),('Chilly Reception','Snow')]:
            r=team();r['fields'][3]=cue
            self.assertEqual(classify(r)[0]['name'],mode)
        for cue,mode in [('Drizzle','Rain'),('Drought','Sun'),('Sand Stream','Sand'),('Sand Spit','Sand'),('Snow Warning','Snow')]:
            r=team();r['fields'][1]=cue
            self.assertEqual(classify(r)[0]['name'],mode)

    def test_every_record_has_cited_field_evidence(self):
        records=json.loads((Path(__file__).parent/'trace.json').read_text())['records']
        for r in records:
            modes=classify(r)
            self.assertEqual(len(modes),len({m['name'] for m in modes}))
            for m in modes:
                self.assertTrue(m['url'].startswith('https://'))
                self.assertTrue(m['evidence'])
                for e in m['evidence']:
                    self.assertTrue(all(0<=c<48 for c in e['columns']))
                    self.assertIn(r['fields'][e['columns'][0]],e['text'])
        counts=Counter(m['name'] for r in records for m in classify(r))
        counts['Unclassified']=sum(not classify(r) for r in records)
        print('Trace mode counts (overlapping):',dict(counts))

if __name__=='__main__':unittest.main()
