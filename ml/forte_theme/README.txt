Forte theme for live Python pages

Inter (local WOFF2, SIL OFL) and the user-supplied Forte logo are embedded.
Morphicons (MIT) animates original outline navigation symbols and the mobile
menu. It is a morphing engine, not an icon artwork collection.
Documentation: https://www.morphicons.com/docs
License copies are kept beside the source. No CDN requests are made by the theme.

Pages: training, scenario, analyze, human_review, human_objects, annotator,
readiness. live_console/page.html is also themed so the periodic status publisher
does not replace the new design with the old one.

After changing theme.css or rebuilding theme.bundle.js, run:
  .venv/bin/python ml/forte_theme/apply.py
This embeds the assets atomically and backs up each previous HTML file. It does
not change API handlers, model processes or human annotations. Do not restart
services just to apply this theme. Existing open review pages are unaffected;
users should save drafts before refreshing to see the design.

To rebuild JS: install the locked package dependencies in this directory and run
  npx esbuild theme.js --bundle --format=iife --minify --outfile=theme.bundle.js
Then run apply.py from the project root.

Mobile navigation follows aria-expanded and aria-current. Animations respect
prefers-reduced-motion. Video pixels, annotation box colors and suspicious-event
colors are not replaced by the brand accent.

Analysis redesign: analysis.css is scoped to body.analysis-page and embedded
after the shared brand stylesheet. The old dark CSS in analyze.html was removed.
The result uses separate media/results/timeline cards; timeline segments reset
button minimum height, borders, padding and radius explicitly. Real completed
job checked at 1440, 1280, 1024, 768 and 390px; threshold/reset/seek verified.

2026-10-18 UI cleanup:
Legacy page CSS colors were migrated to light surfaces at source, including
frame_assist_ui.js and the monitoring publisher template. Shared states use
readable foregrounds. training_archive.html is now included in apply.py.
Root and old /analysis routes redirect into this UI. Next API routes remain.
The competing 3001 release and its iptables switch were retired; see
ml/forte_x/UI_HANDOFF.txt before changing frontend deployment.
Browser checks: seven pages at 1440/390 px, computed text contrast in default
states, training tabs, mobile navigation, root redirects and three read-only APIs.
