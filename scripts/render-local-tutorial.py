# /// script
# requires-python = ">=3.11"
# dependencies = ["Markdown==3.10"]
# ///
"""Build the offline reading copy: uv run --no-project scripts/render-local-tutorial.py."""

import html
import re
from pathlib import Path
from urllib.parse import quote, urlsplit

import markdown


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/getting-started/readme.md"
DESTINATION = ROOT / "docs/local/getting-started.html"
REPO = "https://github.com/jiangshang0725/agentic-rl-lab/blob/feat/local-training-8gb/"


def repository_link(match):
    target = html.unescape(match.group(1))
    parts = urlsplit(target)
    if parts.scheme or target.startswith("#"):
        return match.group(0)
    path = (SOURCE.parent / parts.path).resolve().relative_to(ROOT).as_posix()
    fragment = f"#{quote(parts.fragment)}" if parts.fragment else ""
    return f'href="{html.escape(REPO + quote(path) + fragment, quote=True)}"'


def main():
    source = SOURCE.read_text(encoding="utf-8")
    entries = re.findall(r"^\d+\. \[(.+)\]\(#([a-z]+)\)$", source, re.MULTILINE)
    navigation = "\n".join(
        f'<li><a href="#{anchor}">{i}. {html.escape(label)}</a></li>'
        for i, (label, anchor) in enumerate(entries, 1)
    )
    # Keep one navigable table of contents, with the same stable anchors as Markdown.
    source = re.sub(r"## 目录\n.*?(?=<a id=)", "", source, count=1, flags=re.DOTALL)
    body = markdown.markdown(source, extensions=["fenced_code", "tables", "sane_lists"])
    body = re.sub(r'href="([^"]+)"', repository_link, body)
    style = """
:root{color-scheme:light;--ink:#182832;--muted:#596b75;--accent:#096a70;--line:#dce5e8}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}
body{margin:0;background:#f4f7f8;color:var(--ink);font:16px/1.85 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.shell{max-width:1280px;margin:auto;padding:32px 28px;display:grid;grid-template-columns:248px minmax(0,1fr);gap:40px}
aside{position:sticky;top:28px;align-self:start;max-height:calc(100vh - 56px);overflow:auto;font-size:13px}
.brand{font-weight:750;letter-spacing:.08em;color:var(--accent);margin:0 0 4px}
.edition{font-size:12px;color:var(--muted);margin:0 0 20px}
summary{font-weight:700;cursor:pointer}nav ol{list-style:none;margin:12px 0 0;padding:0}
nav li{margin:0 0 5px}nav a{display:block;padding:4px 8px;border-radius:5px;text-decoration:none;color:var(--muted)}
nav a:hover{background:#e0edef;color:var(--accent)}
main{min-width:0;background:white;border:1px solid var(--line);padding:40px 46px;border-radius:10px}
.tag{display:inline-block;color:var(--accent);background:#e6f3f1;font-size:12px;padding:3px 10px;border-radius:4px;letter-spacing:.06em}
h1{font-size:32px;line-height:1.45;letter-spacing:-.03em;margin:18px 0 22px}
h2{font-size:24px;line-height:1.5;border-top:1px solid var(--line);padding-top:28px;margin:42px 0 18px}
h3{font-size:18px;margin:28px 0 12px}p{margin:14px 0}a{color:var(--accent);text-underline-offset:3px}
strong{font-weight:700}li{margin:6px 0}code{font: .9em/1.7 Consolas,"Cascadia Code",monospace;background:#edf3f5;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}
.code-wrap{position:relative;margin:18px 0}pre{margin:0;background:#142d38;color:#eef6f7;border-radius:7px;padding:42px 18px 18px;overflow:auto}
pre code{display:block;background:none;color:inherit;padding:0;white-space:pre-wrap;overflow-wrap:anywhere;word-break:normal;font-size:13px}
.copy{position:absolute;right:9px;top:8px;border:1px solid #6a838c;border-radius:4px;background:#203f4b;color:#eaf3f5;padding:3px 10px;cursor:pointer;font:12px/1.6 inherit}
.copy:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #bf7846;outline-offset:2px}
table{display:block;overflow:auto;border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}
th,td{border:1px solid var(--line);padding:10px 12px;text-align:left;vertical-align:top;min-width:100px}
th{background:#edf4f5;font-weight:650}tr:nth-child(even){background:#f8fafb}td code{white-space:normal}
footer{margin-top:38px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
@media(max-width:900px){.shell{display:block;padding:18px 14px}aside{position:static;max-height:none;margin:0 0 20px}nav ol{columns:2}.edition{margin-bottom:8px}main{padding:26px 22px}h1{font-size:27px}}
@media(max-width:560px){nav ol{columns:1}body{font-size:15px}main{padding:20px 16px}h1{font-size:25px}h2{font-size:21px}th,td{padding:8px}}
@media print{body{background:white;font-size:10pt}.shell{display:block;padding:0}aside,.copy,.tag{display:none}main{border:0;padding:0}pre{background:#f1f4f6;color:#182832;padding:12px}h2,h3{break-after:avoid}tr{break-inside:avoid}a{color:inherit}table{font-size:9pt}}
"""
    script = """
document.querySelectorAll('pre').forEach(pre => {
  const wrap = document.createElement('div'); wrap.className = 'code-wrap';
  pre.before(wrap); wrap.append(pre);
  const button = document.createElement('button'); button.className = 'copy';
  button.type = 'button'; button.textContent = '复制代码';
  button.addEventListener('click', async () => {
    const text = pre.textContent;
    try {
      if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
      else {
        const field = document.createElement('textarea'); field.value = text;
        field.style.position = 'fixed'; field.style.opacity = '0'; document.body.append(field);
        field.select(); const copied = document.execCommand('copy'); field.remove();
        if (!copied) throw new Error('Clipboard unavailable');
      }
      button.textContent = '已复制';
    } catch (_) { button.textContent = '请选中文本复制'; }
    setTimeout(() => { button.textContent = '复制代码'; }, 2500);
  });
  wrap.append(button);
});
"""
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Windows、RTX 3070 8GB 免费本地强化学习教程：国内镜像、CPU toy、Qwen LoRA、GRPO、SFT、评估与排错。">
<title>本地训练入门 · Agentic RL Lab</title><style>{style}</style></head>
<body><div class="shell"><aside><p class="brand">AGENTIC RL LAB</p><p class="edition">本地学习版 / 2026-11-16</p>
<nav aria-label="教程目录"><details open><summary>学习目录 · 17 节</summary><ol>{navigation}</ol></details></nav></aside>
<main><span class="tag">WINDOWS · RTX 3070 8GB · 无付费 API</span>{body}
<footer>由仓库 experiments/getting-started/readme.md 生成。正文、样式与复制按钮可离线使用；源码和参考链接需要网络。基于 Datawhale Agentic RL Lab。</footer>
</main></div><script>{script}</script></body></html>
"""
    DESTINATION.write_text(document, encoding="utf-8")
    print(f"Wrote {DESTINATION.relative_to(ROOT)} ({len(document.encode('utf-8')):,} bytes)")


if __name__ == "__main__":
    main()
