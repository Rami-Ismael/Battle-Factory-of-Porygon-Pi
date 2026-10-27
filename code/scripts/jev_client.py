"""Jev, TypeSafe AI's decision model, behind the jev_fill.Chooser interface.

JevChooser sends each request through TypeSafe's own SDK (typesafe-sdk 0.7.5, MIT,
github.com/typesafe-ai/typesafe-sdk-python) and returns plain Answer dicts, so the fill loop
cannot tell it from FakeChooser. The model, address and option limit are settings: a later run
could point the same code at another decision model that copies TypeSafe's request shape.

The key comes from TYPESAFE_API_KEY or the macOS Keychain item "typesafe" and is never printed
or written anywhere. Retries are the SDK's own RetryPolicy (timeouts, connection errors and
retryable HTTP statuses, honouring retry-after); any other API error stops the run.

Every call appends one line to the log: the question, the full answer with every probability,
the model name the server reports, its token usage and the time taken. The meta teams are left
out of the log because they are the same in every request; the team and the asked field are kept.
"""
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient, TypeSafeError

import jev_fill

KEYCHAIN_ITEM = 'typesafe'
PRICE_PER_MILLION_INPUT = 0.042     # US dollars, docs.typesafe.ai/models (checked 2026-10-26); output is free


def api_key() -> str:
    key = os.environ.get('TYPESAFE_API_KEY', '').strip()
    if key:
        return key
    found = subprocess.run(['security', 'find-generic-password', '-s', KEYCHAIN_ITEM, '-w'],
                           capture_output=True, text=True)
    if found.returncode or not found.stdout.strip():
        raise SystemExit(f'no TypeSafe key: set TYPESAFE_API_KEY or add the Keychain item "{KEYCHAIN_ITEM}"')
    return found.stdout.strip()


class JevChooser:
    """A jev_fill.Chooser that asks Jev, logs every call and keeps a running token count."""

    def __init__(self, log: Path, model: str = 'jev-latest', base_url: str | None = None,
                 option_limit: int = jev_fill.OPTION_LIMIT, retries: int = 4) -> None:
        self.client = TypeSafeClient(api_key=api_key(), model=model, base_url=base_url,
                                     retry=RetryPolicy(max_retries=retries, backoff_max=20.0))
        self.log, self.option_limit = log, option_limit
        self.usage = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}

    def choose(self, state: dict[str, Any], questions: dict[str, jev_fill.Question]) -> dict[str, jev_fill.Answer]:
        for name, question in questions.items():
            if len(question['criteria']) > self.option_limit:
                raise jev_fill.QuestionTooLarge(f'{name}: {len(question["criteria"])} options, over {self.option_limit}')
        started = time.monotonic()
        sdk_questions = {n: Choice(instructions=q['instructions'], criteria=q['criteria']) for n, q in questions.items()}
        try:
            response = self.client.system_one(state=state, questions=sdk_questions)
        except TypeSafeError as error:      # after the SDK's own retries
            raise jev_fill.TransportError(f'{type(error).__name__}: {error}') from error
        answers: dict[str, jev_fill.Answer] = {}
        for name in questions:
            if name not in response.choices:
                raise ValueError(f'{name}: Jev returned no Choice answer')
            a = response.choices[name]
            answers[name] = {'choice': a.choice, 'confidence': a.confidence, 'probabilities': dict(a.probabilities)}
        usage = response.usage.model_dump() if response.usage else {}
        self.usage['calls'] += 1
        for key in ('input_tokens', 'output_tokens'):
            self.usage[key] += int(usage.get(key) or 0)
        self._write(state, questions, answers, response.model, usage, time.monotonic() - started)
        return answers

    def _write(self, state: dict[str, Any], questions: dict[str, jev_fill.Question], answers: dict[str, jev_fill.Answer],
               model: str, usage: dict[str, Any], seconds: float) -> None:
        record = {'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'model': model, 'seconds': round(seconds, 3), 'usage': usage,
                  'asking_about': state.get('asking_about'), 'team': state.get('team'),
                  'questions': {n: {'instructions': q['instructions'], 'options': list(q['criteria'])} for n, q in questions.items()},
                  'answers': answers}
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def dollars(self) -> float:
        return self.usage['input_tokens'] * PRICE_PER_MILLION_INPUT / 1_000_000

    def close(self) -> None:
        self.client.close()
