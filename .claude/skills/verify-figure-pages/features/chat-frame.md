# Chat frame

A reader sees the page inline in a T3 Code thread, where an agent showed it. The frame is 728 px wide on a desktop and about 360 px on a phone, on the thread's own background. The page must fit it and log nothing.

## Sub-features

- `frame-console`: the page logs no console message inside the frame.
- `frame-layout`: the page drops its own background and side padding, and its content fits the frame's width.

## How to get to it (user POV)

- An agent calls `html_render` with the page's path, and the page appears above the agent's reply.
- The reader scrolls the thread through it, on a desktop or a phone.

## Driving it with html_preview

Preconditions:

- The session is a T3 Code session. Load the tool with ToolSearch: `select:mcp__t3-code__html_preview`.
- The page has been built.

- **Preview dark, desktop width.** Call `mcp__t3-code__html_preview` with `path` set to the page's absolute path, `appearance` `dark` and `width` `728`. The result's `consoleMessages` is `[]`: that is `frame-console`.
- **Read the picture.** In the returned screenshot the title starts at the left edge with no page-coloured band around it, and no element is cut off at the right: that is `frame-layout`.
- **Preview light, and phone width.** Call it again with `appearance` `light`, then with `width` `390`. Same two results each time.
- **Keep the proof.** The tool saves each screenshot and names its path in the result. Record those paths, and the `contentHeight` it reports, with the check ids.

## Gotchas

- The preview captures the page as first drawn. Steps that play on scroll show their starting state below the first screen: an empty reminder panel, empty bars, and "0 of 84 fields hidden". That is the capture, not a fault. The journey in `reader-controls.md` proves those steps.
- `html_preview` checks a page quietly. `html_render` posts it into the thread, so call that only when the user wants the page shown.
- Outside T3 Code the tool does not exist. Report `frame-console` and `frame-layout` as not run.
- The page knows it is framed by the host's `--background` variable. A host that sets none gets the page's own background, and `frame-layout` fails by design.
