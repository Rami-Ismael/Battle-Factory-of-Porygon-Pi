# Reasoning results in Mask Lab

The new section embeds the completed `reasoning-expanded-20261012` audit in
`reasoning-results.json`. The snapshot includes its source SHA-256 and verification
timestamp. It contains aggregates, not API credentials or raw model responses.

Run `python3 build.py` from this directory to regenerate the self-contained page.
The existing snapshot is retained on every build. To deliberately import a newer
completed audit, supply `--reasoning-audit /absolute/path/to/final-audit.json`.
Import rejects missing responses, pending requests, incomplete legal-team scoring,
and nonmatching battle configurations. It never starts generation or battles.

## Comparison boundaries

- The original four-arm dataset stays separate: 5 starters, 270 masks per arm,
  six families and every mask size. Its data, calculations and explorer remain intact.
- The new study has 20 fresh starters, seven largest-mask families, three repeats,
  420 calls per arm. The earlier three-starter reasoning pilot is excluded.
- No trial IDs are joined across studies. Diffusion on the new cohort is explicitly
  unevaluated. The six-setting table reports descriptive rates, never a pooled ranking
  or cross-study treatment difference.
- The matched reasoning battle effect is restricted to 94 both-legal pairs, with
  equal starter weights. Bootstrap intervals resample whole starters. Cached panels
  and their repeated use are reported. All-legal rates use different selected subsets.
- Frozen opponent IDs match across studies and the BC checkpoint fingerprint agrees,
  but seeds and saved policy IDs differ. The older export reports 49 battles and the
  new audit verifies 50. Complete cross-study runtime equivalence is not established.
- Earlier net-per-mask metrics retain the original on rejection. New legality and
  battle metrics do not treat illegal outputs as losses. These estimands are not mixed.

## Verification

The canonical JSON SHA-256 of the original dataset, excluding the newly added
`reasoning` property, remained
`83839204b7a643c0cfb3ed3766cfb929974e7b24da83b2c73c8fdc7320253b57`.
All seven family selections were checked against audit counts; original arm switching
does not alter reasoning results. Chrome checks passed at widths 320, 390, 580 and
1000 with no page overflow or runtime errors. Keyboard access and reduced-motion
behavior were checked; desktop and mobile screenshots were reviewed.
