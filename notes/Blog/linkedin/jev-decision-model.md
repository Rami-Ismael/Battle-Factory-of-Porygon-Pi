Building one Pokémon for a competitive team takes twelve decisions: four rounds to pick the species, then an ability, an item, four moves, a nature and a stat spread. I gave each decision to a decision model, one question at a time. It worked, and it cost about 23 times what a language model charged to write the same Pokémon. Here's why, and the change that should fix most of it.

A decision model is a classifier whose labels you write fresh for every question. I send it the situation and a list of options, here every move this Pokémon can legally learn, and it returns a probability for each one. It can only ever pick from my list. The one I used is Jev, from TypeSafe AI.

Every one of those twelve questions carried the same 18,000-token list of the fifty best teams it was building against. The language model read that list once and wrote the whole Pokémon in one reply. My setup sent it twelve times. The price per token was nearly the same; I was paying to repeat the context.

The fix is in how Jev bills. It reads the shared context once per request, however many questions the request carries. I measured it: two questions about one team cost 20,220 tokens together and 39,524 as two requests. So the next version sends the same step for many teams in one request, with the list of fifty teams read once. I haven't run that yet.

If you use a decision model, count your requests, not just your tokens: the context you repeat is the bill. The full experiments, figures and code are on my project page.
