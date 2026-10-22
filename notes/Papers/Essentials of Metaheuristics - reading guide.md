# Essentials of Metaheuristics: reading guide

Selected on 21 October 2026 for the Pokémon counter-team search project.

**Start here. Read only sections 0.1 and 2.1 in your first sitting.** One candidate solution means one complete legal Pokémon team. A population means a collection of complete teams.

Sean Luke offers the book as [free undergraduate lecture notes](https://people.cs.gmu.edu/~sean/book/metaheuristics/). The [author's book file](https://people.cs.gmu.edu/~sean/book/metaheuristics/Essentials.pdf) identifies the second edition, online version 2.4, dated January 2026. The second print edition is cited as 2013. Section numbers below refer to this edition.

## Your first reading path

| Order | Read | Connect it to your project |
| --- | --- | --- |
| 1 | Section 0.1, What is a Metaheuristic? | Understand the purpose of the search around your team generator. |
| 2 | Section 2.1, Hill-Climbing | Take one team, try nearby changes, and compare their scores. |
| 3 | Chapter 3, Population Methods, opening explanation | Understand why a search keeps or learns from a group of teams. |
| 4 | Section 3.2.1, Crossover and Mutation | Learn rules for producing different candidates. Legal team edits require their own definitions. |
| 5 | Section 3.3.4, Hybrid Optimization Algorithms | Understand memetic algorithms / local search: population search combined with local improvement. |

Follow the chapter 3 introduction into the genetic-algorithm explanation when the terms in section 3.2.1 need context. You do not need to finish every algorithm in the chapter before reading section 3.3.4.

## Keep the reading concrete

After each section, write one Pokémon example and sketch the sequence of steps. For hill-climbing, the example could be a legal move replacement followed by battle evaluation. This is an illustration, not an instruction to run a new experiment.

The book explains general methods. A team that scores higher in a small battle sample may not have a higher true win rate. Use the project's evaluation rules when turning an idea into an experiment.

Continue with [[Introduction to Evolutionary Computing - reading guide]] when you need more detail on representation, legality constraints, or noisy evaluation.

[Back to the reading list](Reading%20List.md#textbooks-and-foundations).
