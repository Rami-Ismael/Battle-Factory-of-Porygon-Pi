"""The Modal redraw split keeps exactly the draws #169's loop makes (inplace_prompt_redraw_modal.py)."""
import inplace_prompt_redraw_modal as M


def row(mask, draw, legal):
    return {'id': mask, 'draw': draw, 'legal': legal}


def test_pending_skips_legal_first_tries_done_draws_and_stopped_masks():
    tasks = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}, {'id': 'd'}]
    first = [row('a', 0, True), row('b', 0, False), row('c', 0, False), row('d', 0, False)]
    done = [row('b', 1, False), row('b', 2, False), row('c', 1, True)]
    assert [(t['id'], d) for t, d in M.pending(tasks, first, done)] == [
        ('b', 3), ('b', 4), ('d', 1), ('d', 2), ('d', 3), ('d', 4)]


def test_loop_keeps_stops_at_first_legal_draw_and_skips_existing():
    existing = [row('b', 1, False), row('c', 2, True)]
    judged = [row('b', 1, False), row('b', 3, True), row('b', 2, False), row('b', 4, True),
              row('c', 3, False), row('c', 1, False), row('d', 1, False), row('d', 2, False)]
    kept = [(r['id'], r['draw']) for r in M.loop_keeps(judged, existing)]
    assert kept == [('b', 2), ('b', 3), ('c', 1), ('d', 1), ('d', 2)]
