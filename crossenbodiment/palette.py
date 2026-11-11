"""palette.py -- the colours the figures and videos share (OOP refactor step 4; step 10 extends it into a Palette).

The three rollout colours of the survey / comparison videos, exactly as scripts/survey_body_videos.py defined them (RGBA 0-1
lists: they go into render_panels specs through json.dumps, so their spelling is part of the output): the reference grey,
z0 red, the searched latent blue (scripts/robot_overlay.py COLORS[0:3] with the grey opaque).
"""

GREY, RED, BLUE = [0.55, 0.57, 0.62, 1.0], [0.85, 0.22, 0.20, 1.0], [0.18, 0.42, 0.85, 1.0]   # robot_overlay.COLORS, grey opaque
