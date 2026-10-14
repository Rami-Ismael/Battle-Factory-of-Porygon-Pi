"""DSPy request structure for the masked-completion baseline. See docs/dspy-structure-experiment.md.

The baseline sends one system string and one JSON user message, and parses the
reply with json.loads. This module keeps the same instructions and the same
input values, and changes only how they are packaged and parsed:

- each input value becomes a named, typed DSPy input field;
- the answer becomes a typed output field, `fills: list[Fill]`, whose JSON schema
  the DSPy adapter shows to the model;
- the reply is parsed by the DSPy adapter into that type.

The parsed answer then goes through the unchanged baseline checks
(`llm_baseline.apply_completion` and the Showdown validator). DSPy only formats
and parses here: no DSPy LM call, retry, adapter fallback or optimization runs.
"""
import json
import re
from typing import Union

import dspy
from pydantic import BaseModel, ConfigDict

KIND = 'dspy'
ADAPTERS = ('chat', 'json')
# Sentences that only dictate the baseline's raw-JSON reply envelope. The DSPy
# adapter replaces that envelope with its own, so these are the only words
# removed. Every other instruction is kept verbatim.
ENVELOPE = (
    'Return only JSON: {"fills":[{"path":[0,"item"],"value":"..."}]}. ',
    ' No prose or Markdown.',
    'Return only strict JSON {"fills":[{"path":[0,"item"],"value":"Leftovers"}]}, adapted to the actual requested paths. ',
    'Use double-quoted keys and strings. No comments, ellipses, trailing commas, Markdown, explanations, or placeholder values. ',
)
# Short labels for the existing user-message keys. They name each value; they
# add no rule or game knowledge that the baseline message lacks.
DESCRIPTIONS = {
    'format': 'Showdown format id',
    'opponents': 'opponent team pastes',
    'masked_team': 'six team slots; null marks a masked field or slot',
    'requested_paths': 'masked paths to fill',
    'legal_species': 'legal species names',
    'regulation_reference': 'regulation reference and legal choices per masked slot',
}
INPUT_TYPES = {'format': str, 'opponents': list[str], 'masked_team': list,
               'requested_paths': list[list[Union[int, str]]], 'legal_species': list[str],
               'regulation_reference': dict}


class _Strict(BaseModel):
    # Strict: "12" is not coerced to 12 and unknown keys are refused, so the
    # typed parse never repairs a value the baseline checks would reject.
    model_config = ConfigDict(strict=True, extra='forbid')


class StatPoints(_Strict):
    HP: int
    Atk: int
    Def: int
    SpA: int
    SpD: int
    Spe: int


class PokemonSet(_Strict):
    species: str
    item: str
    ability: str
    nature: str
    moves: list[str]
    evs: StatPoints


class Fill(_Strict):
    path: list[Union[int, str]]
    value: Union[str, StatPoints, PokemonSet]


def config(adapter, instructions=None, demos=()):
    """Manifest entry for a structured arm. Phase two adds instructions and demos."""
    if adapter not in ADAPTERS:
        raise ValueError(f'unknown DSPy adapter {adapter}; choose from {ADAPTERS}')
    return dict(kind=KIND, adapter=adapter, dspy_version=dspy.__version__,
                instructions=instructions, demos=list(demos))


def adapter_for(structure):
    if structure.get('kind') != KIND:
        raise ValueError(f"unknown request structure {structure.get('kind')}")
    if structure['dspy_version'] != dspy.__version__:
        raise ValueError(f"installed DSPy {dspy.__version__} differs from frozen {structure['dspy_version']}")
    if structure['adapter'] == 'chat':
        # No JSON fallback: the baseline makes no second (repair) call.
        return dspy.ChatAdapter(use_json_adapter_fallback=False)
    if structure['adapter'] == 'json':
        return dspy.JSONAdapter()
    raise ValueError(f"unknown DSPy adapter {structure['adapter']}")


def strip_envelope(system):
    text = system
    for sentence in ENVELOPE:
        text = text.replace(sentence, '')
    if text == system:
        raise ValueError('baseline system text has no known reply-format sentence; add it to ENVELOPE')
    return text.strip()


def baseline_parts(manifest, trial):
    import llm_baseline as B
    plain = {k: v for k, v in manifest.items() if k != 'structure'}
    system, user = B.prompt(plain, trial)
    return system['content'], json.loads(user['content'])


def signature(instructions, keys):
    unknown = set(keys) - set(INPUT_TYPES)
    if unknown:
        raise ValueError(f'no structured type for baseline inputs {sorted(unknown)}')
    fields = {k: (INPUT_TYPES[k], dspy.InputField(desc=DESCRIPTIONS[k])) for k in keys}
    fields['fills'] = (list[Fill], dspy.OutputField(desc='one fill per requested path'))
    return dspy.make_signature(fields, instructions, signature_name='CompleteMaskedTeam')


def parts(manifest, trial):
    """Signature, inputs and adapter that carry the baseline request's content."""
    structure = manifest['structure']
    system, inputs = baseline_parts(manifest, trial)
    instructions = structure.get('instructions') or strip_envelope(system)
    return signature(instructions, list(inputs)), inputs, adapter_for(structure)


def messages(manifest, trial):
    sig, inputs, adapter = parts(manifest, trial)
    return adapter.format(sig, list(manifest['structure'].get('demos') or []), inputs)


def raw_fills(structure, content):
    """The model's own text for the fills field, before any DSPy repair."""
    if structure['adapter'] == 'chat':
        match = re.search(r'\[\[ ## fills ## \]\]\s*(.*?)\s*(?:\[\[ ## |\Z)', content, re.S)
        return match.group(1) if match else None
    try:
        return json.dumps(json.loads(content)['fills'])
    except (ValueError, KeyError, TypeError):
        return None


def parse(manifest, trial, content, diagnostics):
    """Typed parse into the baseline answer shape, {"fills": [...]}.

    `diagnostics['strict_json']` records whether the model's fills text was
    valid JSON as written, so a report can separate DSPy's tolerant JSON
    repair from the model's own output. It is set before parsing can fail.
    """
    if not isinstance(content, str):
        raise TypeError('reply has no answer text')
    sig, _, adapter = parts(manifest, trial)
    try:
        json.loads(raw_fills(manifest['structure'], content))
        diagnostics['strict_json'] = True
    except (TypeError, ValueError):
        diagnostics['strict_json'] = False
    try:
        fields = adapter.parse(sig, content)
    except dspy.AdapterParseError as exc:
        # The baseline counts a ValueError as an invalid answer.
        raise ValueError(f'DSPy parse: {str(exc)[:500]}') from None
    return {'fills': [fill.model_dump() for fill in fields['fills']]}


def section(text, key):
    match = re.search(r'\[\[ ## ' + re.escape(key) + r' ## \]\]\n(.*?)(?=\n\n\[\[ ## |\n\nRespond |\Z)', text, re.S)
    return match.group(1) if match else None


def check_information(manifest, trial):
    """Prove the structured request carries every baseline input value unchanged."""
    _, inputs = baseline_parts(manifest, trial)
    user = messages(manifest, trial)[-1]['content']
    for key, value in inputs.items():
        text = section(user, key)
        if text is None:
            raise ValueError(f'structured request lacks input {key}')
        if (text if isinstance(value, str) else json.loads(text)) != value:
            raise ValueError(f'structured request changed input {key}')
    return sorted(inputs)
