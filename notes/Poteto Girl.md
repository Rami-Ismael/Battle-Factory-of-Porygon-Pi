- [x] Determine an automatic way to go through twitter bookmark to receive the tweet from lauren or know as https://x.com/poteto girl 
- [ ] Read Lauren Tan's [post on trusting agents' work](https://x.com/poteto/status/2111133080526495985) from 2026-10-16, which ranks codebase improvements, [[Static Analysis]], prose and human review as an inverse pyramid.
	- [ ] Go through matpollock skills on codebase improvement first list all the skills that deal with codebase improvement, first define what do they do, afterward create an html with design , i need a subagent that look through is there static analysis tool for pytorch and anything will be useful for machine learning is there a todo list format for proper machine learning, if ther competting choices please make a pro and cons list for me 
		- [ ] go through the artifact The inverse pyramid of agent trust
			- [ ] Go trough the artifact "Codebase improvement kit"
				- [ ] Go through all the high
					- [ ] Prompt the llm how should i prompt claude code to use these skills, then make sure i don't just copy instead give my own spin to it 
						- [x] /improve-codebase-architecture
						- [ ] I split this task into multiples. The first step is to determine what is codebase-desgin, what is the trigger word to call it out. Afterward, determine what is the right place to call codebase-desgin
							- [ ] Write down "“Which modules are too thin?”"
						- [ ] First, i want you to help me understand this skills "domain-modeling" skills, afterward determine how to use the skills because we are gonna use real soon
						- [ ] update the my prompt instruction to use the /retro skill for each claude session that I been throguh 
				- [ ] Static analysis for PyTorch and machine learning code
			- [ ] Read through this section determine what lesson that I can learn from this
	- [ ] Please provide me a clear definition and show me bottom up of why prose are necessary are this software developement in the modern era because of llm 
	- [x] Open the [pstack source in cursor/plugins](https://github.com/cursor/plugins/tree/main/pstack) that the post says to install.
- [x] Read Lauren Tan's [pstack ten-second tutorial](https://x.com/poteto/status/2110757066709356851) from 2026-10-15, which covers /setup-pstack, /poteto-mode and /poteto-help.
	- [x] Open the [pstack page on the Cursor marketplace](https://cursor.com/marketplace/cursor/pstack).
	- [x] Open the [pstack plugin for Grok Bot](https://x.ai/bot/plugin/9717366).
- [x] Read Lauren Tan's [post on uses for Grok Bot](https://x.com/poteto/status/2109326308950032501) from 2026-10-11, which links a thread of almost a thousand ideas.
- [ ] Read Lauren Tan's [The Complete Guide to pstack Pt. 2](https://x.com/poteto/status/2097732320606507506) from 2026-09-09, an X Article on supervising an agent that is smarter than you.
	- [ ] Read [Part 1 of the guide](https://x.com/poteto/status/2094457600259842065) first, since Part 2 builds on its verification skills.
	- [ ] Compare the article's [/why skill](https://github.com/cursor/plugins/blob/main/pstack/skills/why/SKILL.md), [/recall skill](https://github.com/cursor/plugins/blob/main/pstack/skills/recall/SKILL.md), [/technical-writing skill](https://github.com/cursor/plugins/blob/main/pstack/skills/technical-writing/SKILL.md) and [/unslop skill](https://github.com/cursor/plugins/blob/main/pstack/skills/unslop/SKILL.md) with your pskill copies.
	- [ ] Read the [README section on why pstack has no planning skills](https://github.com/cursor/plugins/blob/main/pstack/README.md#why-are-there-no-planning-skills).
	- [ ] Decide whether the [prototyping playbook](https://github.com/cursor/plugins/blob/main/pstack/skills/poteto-mode/playbooks/prototype.md) and the [multi-phase planning playbook](https://github.com/cursor/plugins/blob/main/pstack/skills/poteto-mode/playbooks/multi-phase-plan.md) are worth porting to pskill.
- [ ] https://x.com/poteto/status/2094457600259842065

# Advice from @poteto (Lauren Tan) for Working with Claude / Coding Agents

Practical checklist drawn from Lauren Tan’s (@poteto) public posts and skills (pstack / poteto-mode). She works on Grok Bot at SpaceXAI.

## Core principles

- [ ] **Prioritize the environment over micromanaging the agent**  
  Fix the codebase, types, architecture, and checks so the agent does the right thing by default instead of constantly correcting it.  
  → https://x.com/poteto/status/2106542593656111276  
  → https://x.com/poteto/status/2111133080526495985

- [ ] **Use an inverse pyramid of trust**  
  Put most effort into deterministic layers (code architecture + static analysis) and the least into human review of every change.  
  → https://x.com/poteto/status/2111133080526495985

- [ ] **Make the easy path the correct path**  
  Agents copy existing patterns and take the shortest route. Design the codebase so the shortest path is high-quality.  
  → https://x.com/poteto/status/2111133080526495985  
  (“agents love to copy what exists and take the shortest path, so make the easy thing the right thing”)

- [ ] **Prefer code & checks over prose**  
  Comments, AGENTS.md, and free-form rules are hard to enforce and go stale. Encode constraints in types, architecture, lint rules, and tests instead.  
  → https://x.com/poteto/status/2111133080526495985  
  → https://x.com/poteto/status/2110758788525621282

- [ ] **Delete most code comments**  
  Agents treat comments as permission to keep workarounds. Keep only the rare ones that explain non-obvious external quirks; otherwise solve the underlying issue.  
  → https://x.com/poteto/status/2110149626133438732  
  → https://x.com/poteto/status/2110758788525621282

- [ ] **Build trust before running long autonomous loops**  
  A low-trust loop just wastes tokens and compounds “slop.” Invest in verification and environment quality first.  
  → https://x.com/poteto/status/2063815573361111470

- [ ] **Verify against the real running system**  
  “It compiles” or “the tests pass” is not enough. Exercise the actual feature and gather evidence. (Also reflected in her emphasis on verification skills and self-checking loops.)  
  → https://x.com/poteto/status/2063815573361111470  
  → https://x.com/poteto/status/2111596610836955561 (working on a new verification skill)

- [ ] **Turn repeated corrections into durable constraints**  
  When you keep fixing the same agent mistake, encode the fix in architecture, types, or automated checks (e.g. via `/correct`).  
  → https://x.com/poteto/status/2106542593656111276

- [ ] **Use adversarial / multi-perspective review**  
  Have specialized reviewers attack the change from architecture, correctness, and simplicity angles (supported by her skill ecosystem and pstack playbooks; adversarial-review is a known poteto-associated skill pattern).  
  → https://x.com/poteto/status/2110757066709356851 (pstack as rigorous skill set)

- [ ] **Keep prose clean (“unslop”)**  
  Strip verbose boilerplate, fake robustness, and unnecessary abstractions from both code and agent replies.  
  → https://x.com/poteto/status/2090582845056336270  
  → https://x.com/poteto/status/2063815573361111470 (slop compounds without engineering taste)

- [ ] **Be deliberate with sub-agents**  
  Spawn them intentionally and review their work (reflected in poteto-mode / pstack playbooks that route work deliberately).  
  → https://x.com/poteto/status/2110757066709356851

- [ ] **Bias toward the smallest change that solves the problem**  
  Prefer deletion and simplicity over new layers (consistent with her “make the easy thing the right thing” and environment-first framing).  
  → https://x.com/poteto/status/2111133080526495985

- [ ] **Prototype to answer open questions**  
  When the agent could settle an empirical question by trying something, let it prototype instead of asking you.  
  → https://x.com/poteto/status/2111133080526495985  
  → https://x.com/poteto/status/2106542593656111276  
  (“answer open questions with prototyping rather than defer to me”)

- [ ] **Install and use structured skills (pstack / poteto-mode)**  
  Prefix non-trivial work with `/poteto-mode` so the agent follows rigorous playbooks.  
  → https://x.com/poteto/status/2110757066709356851

## Quick setup tips

- [ ] Install her skill stack (pstack or a Claude Code port of it).  
  → https://x.com/poteto/status/2110757066709356851

- [ ] Run `/setup-pstack` (or the Claude equivalent) and choose a budget.  
  → https://x.com/poteto/status/2110757066709356851  
  → https://x.com/poteto/status/2110534523389943974

- [ ] Use `/poteto-help` when you’re unsure which skill to invoke.  
  → https://x.com/poteto/status/2110757066709356851  
  → https://x.com/poteto/status/2110760668987322408

- [ ] Put `/poteto-mode` in front of any prompt you want done rigorously.  
  → https://x.com/poteto/status/2110757066709356851

- [ ] Create verification skills so agents can check their own work.  
  → https://x.com/poteto/status/2111133080526495985  
  → https://x.com/poteto/status/2111596610836955561

# Pskill

1. Swarm make sense if you have lots of agents