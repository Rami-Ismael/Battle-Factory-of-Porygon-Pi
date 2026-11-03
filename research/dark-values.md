# How do published color systems make dark values from light ones?

Read 2026-11-03 from primary docs. No recommendation.

**None of the readable pages says whether dark values are tuned by hand or computed.** What they show:

| | Radix | Primer | Carbon |
|---|---|---|---|
| Dark set | Separate files, such as `gray-dark.css` | Neutral scale direction is inverted, so themes can share functional tokens | Same token names, with different values per theme |
| Backgrounds | Warns against saturated greys for app backgrounds, "especially in dark mode" | First six neutral steps are typically backgrounds | Gray 100 `#161616` and Gray 90 `#262626`. Layers get one step lighter |
| Accents in dark | Not on the pages read | Not on the page read | No dark-specific change stated |
| Charts in dark | Not on the pages read | No dark-specific chart colors found | Sequential: the lightest color marks the largest values. Diverging: no difference between themes. Categorical: dark values not readable from the page text |
| Dark-only contrast rule | None | None. Chart marks need 3:1 in general | None. 4.5:1 for small text and 3:1 for graphics in general |

Pattern: all three ship dark as its own set of values under shared names. None has a separate contrast rule for dark, so the WCAG 2.2 numbers apply in both themes.

Sources:

- [Radix Colors, composing a palette](https://www.radix-ui.com/colors/docs/palette-composition/composing-a-palette)
- [Radix Colors, usage](https://www.radix-ui.com/colors/docs/overview/usage)
- [Primer, color usage](https://primer.style/product/getting-started/foundations/color-usage/)
- [Primer, data visualization](https://primer.style/product/ui-patterns/data-visualization/)
- [Carbon, color](https://carbondesignsystem.com/building-blocks/foundations/color)
- [Carbon, data visualization color palettes](https://carbondesignsystem.com/data-visualization/color-palettes/)

Gaps: Material's dark theme page returned no text, so Material isn't covered. Carbon's themes page and its categorical dark values weren't read.
