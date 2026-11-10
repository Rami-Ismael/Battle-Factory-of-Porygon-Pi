# SPEC: Nine tries, one π

GOAL:    the viewer remembers that the logo won an elimination: nine real designs, one survived, and it still reads at 16 px.
OUTPUT:  HyperFrames, 1080 x 1350 (4:5 feed), 30 fps
TIME:    32.6 s x 30 = 978 frames; no music (HeyGen not signed in), bundled sound effects only
LOOK:    ground #E9E8E4 flat, no texture; cards are the boards' own paper #FAFAFA with their grey construction lines;
         ink #111111; accent only on the survivor: pink #E8607F bar, blue #2F8FC9 legs (the mark's own colours);
         display Outfit 700, 76 px, -0.02em, line height 1.05, at most 2 lines, left edge x = 108, top y = 150;
         labels Outfit 400, 36 px, #555555; no shadows, no glow, no grain.
OBJECTS: the nine real construction boards (brand/logo/0*/board*.png, centre 1440 px square crop), the real
         exports icon-512.png, favicon-32.png, favicon-16.png, mark-color.svg; one generic browser tab strip.
MOTION:  arrivals expo.out 0.3-0.45 s; moves power3.inOut 0.8-1.0 s; exits power3.in, 40-50 % of the entrance;
         strike line scaleX 0 -> 1 in 0.28 s power2.out at -8 deg (registry strikethrough-replace recipe);
         title entrances vary: per-word blur 6 px -> 0 (0.13 s, 0.067 s apart), per-word grey -> ink, whole-line blur;
         titles hold dead still once landed.
SHOTS:
  0.0-4.2    3 x 3 wall of the nine boards (290 px cards, 16 px gaps); reads "9 logos. 1 survives."  Frame 0 is the thumbnail.
  4.2-7.4    01 and 02 dim to 22 % (no recorded reason, so no strike); "The first two had no π". clicks.
  7.4-10.6   03, 05, 06 struck and dimmed; "Facets beat tiles, hexagon and bird". three clicks, rotated takes.
  10.6-13.6  the five losers drop out; the four faceted πs regroup into a 2 x 2 (420 px); "Four faceted πs left". whoosh.
  13.6-16.4  04 struck; "No feet: it reads 77". click.
  16.4-19.4  04c and 04d struck; "Short legs read as bunting". two clicks.
  19.4-22.4  04b travels to centre and grows to 900 px (registry parallax-zoom); its grid blurs away into the colour
             mark; "Two feet, 25 facets". whoosh.
  22.4-25.9  the mark shrinks into a row: 512 px export, 32 px and 16 px favicons at 4x, pixels visible;
             "At 16 px the gaps close". pops.
  25.9-28.6  a browser tab strip with the 16 px favicon; "Still a π in a browser tab". short whoosh.
  28.6-32.6  end card: the favicon grows out of the tab into the colour mark (300 px wide), bass impact on landing,
             "Battle Factory of Porygon-Pi" wipes in under it and holds 2 s+.
BANNED:  dark void; glow, particles; cream-paper + serif + mono-caps-label look; everything fading and sliding up;
         three cards in a row; docked logo; invented reasons for 01, 02, 03, 05, 06; a drawn copy of any board.
QA:      0.0 (thumbnail: nine boards and the title), 8.4 (three strikes), 12.0 (2 x 2), 18.5 (only 04b clean),
         20.6 (seam: 04b at 900 px), 24.5 (size row), 27.5 (tab), 32.5 (end card).
