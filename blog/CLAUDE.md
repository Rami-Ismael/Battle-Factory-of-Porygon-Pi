# blog/

The main blog article for Battle Factory of Porygon-Pi: how Rami made the cross-entropy method work for building Pokémon VGC teams. It is one article, at `beating-the-meta-pokemon-vgc/index.html`, served by this repository's GitHub Pages site at `https://rami-ismael.github.io/Battle-Factory-of-Porygon-Pi/blog/beating-the-meta-pokemon-vgc/`. The daily posts in `notes/Blog/` are a separate working log.

## This folder is the source

`notes/` and `code/` are mirrors: `tools/sync.py` overwrites them from Rami's laptop. `blog/` is written here. Edit it in place, commit on a branch, and open a pull request.

## Rami writes the prose

Build the layout, the markup and the figure embeds. The paragraphs are his: when a section needs text, add the heading he asked for and leave the body empty. To reword something he wrote, quote it and propose the replacement in chat.

## Reader

ML-curious engineers and students. The article is a worked ML example by someone who plays Pokémon Champions regularly, so it explains each ML idea from scratch, includes a short VGC primer, and puts the main result early.

## Layout

Distill style, in plain HTML and CSS with no build step:

- one reading column (`.col`, 42rem);
- figures that break out of the column, each with a numbered caption (`figure`, `figcaption`);
- margin notes (`aside.note`), in the right gutter on wide screens and inline on a phone.

Fonts and colour tokens match `notes/Blog/visuals/repo-sync/repo-sync.html`. They live in the `:root` block of the article; take every colour from those tokens so both themes keep working.

## Figures

Figures are built in the Obsidian vault, because their build scripts read data that exists only on the laptop. They reach this repository in `notes/Blog/visuals/` through the sync. The article loads each one by relative path, `../../notes/Blog/visuals/<figure>/<page>.html`, so there is one copy of every figure. To change a figure, change it in the vault.

## Publishing

The article is marked as a draft in two places in `index.html`: the `noindex` meta tag and the "Draft" label in the byline. Publishing means removing both. The `og:image` is already a 1200 × 630 still, `notes/Blog/visuals/noise-to-archetypes/preview.png`.
