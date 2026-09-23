#!/usr/bin/env bash
# Download the paper and regenerate the local text extracts (not committed - copyright).
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE=17569229
for i in $(seq 1 20); do
  have=$(stat -c %s paper_2608.10744.pdf 2>/dev/null || echo 0)
  [ "$have" -ge "$SIZE" ] && break
  curl -sL -C - --max-time 300 -o paper_2608.10744.pdf "https://arxiv.org/pdf/2608.10744v1" || true
done
pdftotext -layout paper_2608.10744.pdf paper_main_text.txt
curl -sL --max-time 120 -o /tmp/paper.html "https://arxiv.org/html/2608.10744v1"
python3 - <<'PY'
import html, re
src = open('/tmp/paper.html', encoding='utf-8', errors='ignore').read()
src = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', src, flags=re.S)
src = re.sub(r'<math[^>]*alttext="([^"]*)"[^>]*>.*?</math>', lambda m: ' $' + html.unescape(m.group(1)) + '$ ', src, flags=re.S)
src = re.sub(r'</(p|div|h1|h2|h3|h4|li|tr|section)>', '\n', src)
src = re.sub(r'</t[dh]>', ' | ', src)
src = html.unescape(re.sub(r'<[^>]+>', '', src))
src = re.sub(r'[ \t]+', ' ', src)
open('paper_html_text.txt', 'w', encoding='utf-8').write(re.sub(r'\n\s*\n+', '\n', src))
PY
echo "paper + text extracts ready"
