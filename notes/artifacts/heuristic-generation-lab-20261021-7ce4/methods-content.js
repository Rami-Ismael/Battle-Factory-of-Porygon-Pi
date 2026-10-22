window.METHODS = [
  {
    id: "gp",
    label: "Genetic programming",
    shortLabel: "GP",
    tagline: "Evolve the rule as a tree.",
    intro: "Treat each formula or small program as an individual. Evaluate the decisions it causes, keep effective individuals, and mutate or combine their parts.",
    representation: "An expression tree whose leaves are measured features or constants and whose branches are permitted operations.",
    provide: [
      "Features the rule can read",
      "Typed operations, such as addition, minimum, comparison, and conditional choice",
      "An evaluator and limits on tree size and depth"
    ],
    steps: [
      { title: "Build trees", text: "Generate candidate expressions from your permitted features and operations." },
      { title: "Test decisions", text: "Compile each tree into a callable rule and measure the resulting search outcomes." },
      { title: "Evolve parts", text: "Select effective trees, replace subtrees, and combine parent subtrees." }
    ],
    example: {
      before: "score = gain",
      after: "score = max(gain, coverage) - cost",
      caption: "Illustrative mutation. Whether the added terms help must be measured."
    },
    bestFor: "Inspectable formulas and small rules with a deliberately restricted vocabulary.",
    limitation: "The allowed operations restrict discoveries. Symbolic regression fitted to observed win rates produces a predictor; decision fitness evaluates how a rule changes search outcomes.",
    startHere: { label: "DEAP typed genetic programming tutorial", url: "https://deap.readthedocs.io/en/stable/tutorials/advanced/gp.html" },
    sources: [
      { label: "DEAP genetic programming", url: "https://deap.readthedocs.io/en/stable/tutorials/advanced/gp.html" }
    ]
  },
  {
    id: "grammar",
    label: "Grammatical evolution",
    shortLabel: "Grammar",
    tagline: "Define a language, then search its programs.",
    intro: "Write a grammar describing legal rule structures. Evolution changes a representation that maps through this grammar into a program.",
    representation: "A genome mapped to formulas, conditional rules, or operator sequences through a formal grammar.",
    provide: [
      "Grammar rules and available features",
      "A task-specific fitness evaluator",
      "Search settings and runtime checks"
    ],
    steps: [
      { title: "Define syntax", text: "Specify the expressions and branches a generated rule may contain." },
      { title: "Map and run", text: "Map each genome into a program, then assess its behavior." },
      { title: "Evolve genomes", text: "Retain effective candidates and vary the representations that produced them." }
    ],
    example: {
      before: "<rule> ::= score(<feature>)",
      after: "<rule> ::= if <condition> then <rule> else <rule>",
      caption: "Adding a grammar production permits branching. Evolution searches within the language you supply."
    },
    bestFor: "Controlling the structure of generated programs, including conditions and sequences.",
    limitation: "A grammar containing only linear sums cannot discover arbitrary branching algorithms. Syntactic validity does not guarantee useful or numerically valid behavior.",
    startHere: { label: "PonyGE2 implementation", url: "https://github.com/PonyGE/PonyGE2" },
    sources: [
      { label: "PonyGE2 paper", url: "https://arxiv.org/abs/1703.08535" },
      { label: "PonyGE2 source", url: "https://github.com/PonyGE/PonyGE2" }
    ]
  },
  {
    id: "llm",
    label: "LLM code evolution",
    shortLabel: "LLM",
    tagline: "Generate code, execute it, and use the results.",
    intro: "An LLM proposes heuristic functions. An evaluator measures their behavior, and evolutionary search uses effective candidates to guide further proposals.",
    representation: "Executable functions, sometimes accompanied by natural-language ideas or feedback.",
    provide: [
      "A precise function signature and permitted inputs",
      "Problem description and seed code",
      "An evaluator, LLM access, and isolated execution"
    ],
    steps: [
      { title: "Propose", text: "Generate a candidate function inside a fixed search procedure." },
      { title: "Measure", text: "Reject invalid outputs and score completed search outcomes." },
      { title: "Revise", text: "Use selected programs and measured feedback to request new candidates." }
    ],
    example: {
      before: "def rank(features):\n    return features.gain",
      after: "def rank(features):\n    return max(features.gain, features.coverage) - features.cost",
      caption: "A generated function is a proposal. Execution and comparison provide the evidence."
    },
    bestFor: "Exploring code beyond a small hand-built grammar while retaining readable functions.",
    limitation: "Generation and evaluation both cost resources. Published results on other tasks establish no Pokémon win-rate improvement. A timeout alone does not isolate generated code.",
    startHere: { label: "Adapt EoH to a problem", url: "https://github.com/FeiLiu36/EoH" },
    variants: [
      {
        name: "FunSearch",
        idea: "Keep a skeleton fixed and evolve a selected function.",
        implementation: "Official code supplies pipeline components but excludes the LLM backend, sandbox, and distributed infrastructure.",
        source: "https://doi.org/10.1038/s41586-023-06924-6",
        repo: "https://github.com/google-deepmind/funsearch"
      },
      {
        name: "EoH",
        idea: "Evolve a written heuristic idea together with its code.",
        implementation: "Supply a problem class, evaluator, and remote or local LLM configuration.",
        source: "https://proceedings.mlr.press/v235/liu24bs.html",
        repo: "https://github.com/FeiLiu36/EoH"
      },
      {
        name: "ReEvo",
        idea: "Use short-term and accumulated reflections to guide crossover and mutation.",
        implementation: "Add problem configuration, evaluation, function descriptions, signature, and seed function.",
        source: "https://proceedings.neurips.cc/paper_files/paper/2024/file/4ced59d480e07d290b6f29fc8798f195-Paper-Conference.pdf",
        repo: "https://github.com/ai4co/reevo"
      }
    ],
    sources: [
      { label: "EoH paper", url: "https://proceedings.mlr.press/v235/liu24bs.html" },
      { label: "ReEvo source", url: "https://github.com/ai4co/reevo" }
    ]
  },
  {
    id: "neural",
    label: "Neural and reinforcement learning",
    shortLabel: "Neural / RL",
    tagline: "Learn the decision function through training.",
    intro: "Train a model to choose construction or improvement actions using measured rewards. The reusable heuristic can be the trained model and its inference procedure.",
    representation: "A learned policy, or an explicit rule generator if program output is part of its design.",
    provide: [
      "A representation of search state",
      "Available actions and reward",
      "Training instances and repeated training evaluations"
    ],
    steps: [
      { title: "Observe", text: "Encode the information available at each decision." },
      { title: "Act and learn", text: "Choose actions and update the model using their rewards." },
      { title: "Deploy", text: "Use the trained policy on untouched cases and measure its outcomes." }
    ],
    example: {
      before: "choose uniformly from legal edits",
      after: "policy(search_state) → probabilities over legal edits",
      caption: "The learned model becomes the rule. It need not emit readable source code."
    },
    bestFor: "Repeated decision problems where you can afford training and accept a learned model.",
    limitation: "Direct solution generation is not automatically a classical generation hyper-heuristic. Choosing existing operators is selection; emitting programs requires an explicit program language.",
    startHere: { label: "Attention model authors' implementation", url: "https://github.com/wouterkool/attention-learn-to-route" },
    sources: [
      { label: "Attention, Learn to Solve Routing Problems!", url: "https://arxiv.org/abs/1803.08475" }
    ]
  },
  {
    id: "config",
    label: "Algorithm configuration",
    shortLabel: "Configuration",
    tagline: "Search the settings of an existing rule.",
    intro: "When the rule structure is already useful, search its weights, thresholds, or predefined component choices before generating new code.",
    representation: "A vector of settings or categorical choices within a supplied algorithm family.",
    provide: [
      "An existing algorithm",
      "Permitted settings and ranges",
      "Training instances and an evaluator"
    ],
    steps: [
      { title: "Expose settings", text: "Define which values or components may change." },
      { title: "Compare", text: "Evaluate configurations across the training instances." },
      { title: "Validate", text: "Check the selected settings on untouched cases." }
    ],
    example: {
      before: "score = 1.0 * gain - 1.0 * cost",
      after: "score = 2.4 * gain - 0.6 * cost",
      caption: "Illustrative values. The formula structure stays fixed while its settings change."
    },
    bestFor: "A strong first comparison when weights or thresholds are uncertain.",
    limitation: "Fixed-structure tuning does not generate new code. Component configuration cannot invent components outside the supplied space.",
    startHere: { label: "irace configuration guide", url: "https://iridia.ulb.ac.be/irace/README.html" },
    sources: [
      { label: "irace documentation", url: "https://iridia.ulb.ac.be/irace/README.html" }
    ]
  }
];
