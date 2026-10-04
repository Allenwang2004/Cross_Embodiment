# Project page

Static page for GitHub Pages. `index.html` is generated from `page.html`
(the same content without the document wrapper) — edit `page.html`, then rebuild:

    bash build_index.sh

Deploy: push this folder's contents to a `gh-pages` branch (or to the root of a
`<user>.github.io` repository) and enable Pages in the repository settings.
Before publishing: replace "Author names" / "Affiliation" in the header, and make
sure the Code / Full report links point to a public location.
