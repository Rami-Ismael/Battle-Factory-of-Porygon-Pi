# What names do published color systems give greys, accents and chart colors?

Read 2026-11-03 from primary docs. No recommendation.

| Role | Radix | Primer | Carbon |
|---|---|---|---|
| Greys | Numbered scale: `--gray-1` to `--gray-12`. Surfaces by role: `--color-background`, `--color-panel-solid` | By role: `--fgColor-default`, `--fgColor-muted`, `--bgColor-default` | By role: `$background`, `$layer-01`, `$text-primary`, `$text-secondary` |
| Accent | `--accent-1` to `--accent-12` | `--fgColor-accent`, `--bgColor-accent-emphasis`, `--bgColor-accent-muted` | No token called accent. `$interactive`, `$button-primary`, `$link-primary` |
| Text on a solid accent | `--accent-contrast` | `--fgColor-onEmphasis` | `$text-on-color` |
| Chart colors | Not on the pages read | Separate: `--data-blue-color-emphasis`, `--data-blue-color-muted` | Separate palettes named by hue and step, such as Purple 70, not tokens |
| Status colors | No status tokens on the pages read. Red and Amber are accent scales | Separate: `--fgColor-danger`, `--fgColor-success`, `--fgColor-attention` | Separate: `$support-error`, `$support-warning`, plus an Alert palette for charts |

Patterns:

- Two of the three name greys by role. Radix numbers its steps, and adds role names for surfaces.
- All three have a dedicated name for text on a solid color.
- Primer and Carbon keep status colors apart from chart colors.
- Primer's chart names are the closest to the draft's `--data-1`, but they use the hue, not a number.

Sources:

- [Radix Colors, understanding the scale](https://www.radix-ui.com/colors/docs/palette-composition/understanding-the-scale)
- [Radix Themes, color](https://www.radix-ui.com/themes/docs/theme/color)
- [Primer, color primitives](https://primer.style/product/primitives/color/)
- [Carbon, color tokens](https://carbondesignsystem.com/elements/color/tokens/)
- [Carbon, data visualization color palettes](https://carbondesignsystem.com/data-visualization/color-palettes/)

Limit: a small page-reading agent read these pages. I re-checked `--fgColor-onEmphasis`, `--data-blue-color-emphasis` and `$text-on-color` against the pages, but not every token name.
