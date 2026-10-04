#!/usr/bin/env bash
# Rebuild the standalone GitHub Pages files from the two page sources:
#   page.html    -> index.html  (English)
#   page_zh.html -> zh.html     (繁體中文)
# The <!--LANG--> slot becomes a language switch only here, so the claude.ai
# previews (published from page*.html) carry no link to a file they don't have.
cd "$(dirname "$0")"
build () {  # build <src> <out> <lang> <switch-html>
  { printf '<!doctype html>\n<html lang="%s">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n<style>:root{color-scheme:light}body{margin:0}img{max-width:100%%}[hidden]{display:none!important}</style>\n' "$3"
    sed -n '1,/^<\/style>/p' "$1"; printf '</head>\n<body>\n'
    sed -n '/^<\/style>/,$p' "$1" | tail -n +2 | sed "s#<!--LANG-->#$4#"
    printf '</body>\n</html>\n'; } > "$2"
  echo "-> $2"
}
build page.html    index.html en    '<div class="lang">English · <a href="zh.html">中文</a></div>'
build page_zh.html zh.html    zh-Hant '<div class="lang"><a href="index.html">English</a> · 中文</div>'
