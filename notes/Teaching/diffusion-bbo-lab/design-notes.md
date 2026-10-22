# Diffusion-BBO visual lesson

Purpose: explain inverse generation, the paper's norm-based ensemble estimator, target selection, and the evaluation loop through a clearly labeled toy experiment.

Palette: ice #edf4fb; paper #ffffff; ink #20334b; blue #2457ce; violet #7957ad; teal #087f87. Five lane identities add ochre and berry; labels also identify each lane.

Type: Avenir Next (system fallback Trebuchet MS) for navigation and explanation; Georgia for the large question. Formulae use the body face with readable Unicode symbols.

Layout candidates:

    A: [intro] [five equal cards] [formula]
    B: [question + purpose]
       [one shared ruler with five model lanes | controls]
       [target comparison | actual evaluation]
       [representation limitation]

Choose B. Reject identical cards because they hide the shared measurement scale and suggest unrelated models. The central instrument, not ornamental chrome, provides the visual identity. Text stays left aligned; quantities align by column.

Motion gate: occasional educational interaction; purpose is state indication and explanation. CSS transform/opacity transitions with cubic-bezier(0.77,0,0.175,1), 800ms for the deliberately slow explanatory sample movement; 200ms opacity for state feedback. Steps remain independently readable. Replay is user-triggered, stops after one pass, and can be interrupted. Keyboard state changes and reduced-motion mode skip spatial motion. No libraries or network assets required.

Scientific boundaries: points are synthetic one-dimensional positive design coordinates, hence their norms equal their values. Within-model spread and between-model mean variance are controllable independently. Population variance is used for this illustration. The target-table penalties and evaluated scores are invented, not paper results or Pokémon predictions. Norm agreement does not imply distribution agreement or reliable objective values.

Sources: arXiv 2407.00610v3, Sections 4–5 and Appendix D.2; official NeurIPS 2024 BDU workshop listing. Source-derived claims are kept concise; extended examples are original synthetic illustrations.
