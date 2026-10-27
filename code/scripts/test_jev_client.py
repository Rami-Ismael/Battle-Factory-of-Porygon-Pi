"""The Jev adapter's error mapping, without the network or a key."""
import unittest

import httpx2
from typesafe_sdk._core.errors import TypeSafeAPIError

import jev_client
import jev_fill

QUESTION: jev_fill.Question = {'type': 'choice', 'instructions': 'Which?', 'criteria': {'a': None, 'b': None}}


class Refusing:
    def __init__(self, status: int) -> None:
        self.status = status

    def system_one(self, **_kwargs: object) -> None:
        raise TypeSafeAPIError(self.status, {}, httpx2.Headers(), 'refused')


def chooser(status: int) -> jev_client.JevChooser:
    c = jev_client.JevChooser.__new__(jev_client.JevChooser)    # no key: the client is replaced below
    c.client, c.option_limit = Refusing(status), jev_fill.OPTION_LIMIT
    c.usage = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}
    return c


class ErrorMappingTest(unittest.TestCase):
    def test_no_credits_stops_the_run(self):
        with self.assertRaises(jev_client.OutOfCredits):
            chooser(402).choose({}, {'q': QUESTION})

    def test_other_api_errors_are_retried_later(self):
        with self.assertRaises(jev_fill.TransportError):
            chooser(500).choose({}, {'q': QUESTION})


if __name__ == '__main__':
    unittest.main()
