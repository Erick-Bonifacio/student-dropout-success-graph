"""Monta os slides a partir de slides/fonte_apresentacao.html.

Embute as figuras de figuras/slides/ (geradas por figuras_slides.py) e os números do painel
da demonstração, e grava:
  slides/apresentacao_grafos.html   o deck completo
  slides/demonstracao.html          só o slide da demonstração

Uso (na raiz do repositório):
  python figuras_slides.py
  python slides/montar_slides.py
O PDF sai do próprio HTML: abrir no Chrome, Ctrl+P, "Salvar como PDF" (um slide por página).
"""
import base64
import io
import pathlib
import re

from PIL import Image

root = pathlib.Path(__file__).resolve().parent.parent
s = (root / "slides/fonte_apresentacao.html").read_text(encoding="utf-8")


def b64(nome):
    # resolução cheia (2x do slide); 256 cores mantêm as bordas nítidas e reduzem o tamanho
    im = Image.open(root / f"figuras/slides/{nome}.png").convert("RGB")
    buf = io.BytesIO()
    im.quantize(256).save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


s = re.sub(r"\{\{IMG:([\w]+)\}\}", lambda m: b64(m.group(1)), s)
s = re.sub(r"\{\{FILE:([\w.]+)\}\}", lambda m: (root / f"figuras/slides/{m.group(1)}").read_text(encoding="utf-8"), s)
assert "{{" not in s
(root / "slides/apresentacao_grafos.html").write_text(s, encoding="utf-8")
print("apresentacao_grafos.html:", len(s) // 1024, "KB")

ini = s.index('<div id="viewport"><div id="stage">') + len('<div id="viewport"><div id="stage">')
fim = s.index('</div></div>\n<div id="prog">')
sec = re.search(r'<section class="slide" data-title="Demonstração".*?</section>', s, re.S).group(0)
d = s[:ini] + "\n\n" + sec + "\n\n" + s[fim:]
d = d.replace("<title>Evasão em Grafos</title>", "<title>Demonstração do Protótipo</title>", 1)
(root / "slides/demonstracao.html").write_text(d, encoding="utf-8")
print("demonstracao.html:", len(d) // 1024, "KB")
