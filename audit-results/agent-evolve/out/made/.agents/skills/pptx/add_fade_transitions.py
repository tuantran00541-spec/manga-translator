"""add_fade_transitions.py — post-save patch: add a uniform fade transition to
every slide in a saved .pptx (python-pptx has no transition API).
Usage: python add_fade_transitions.py deck.pptx [out.pptx]"""
import shutil, sys, zipfile, re

FADE = '<p:transition xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" spd="med"><p:fade/></p:transition>'
NS_P = 'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'

def main(src, dst=None):
    dst = dst or src
    zin = zipfile.ZipFile(src)
    entries = zin.namelist()
    slide_files = [n for n in entries if re.match(r"ppt/slides/slide\d+\.xml$", n)]
    out = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
    patched = 0
    for name in entries:
        data = zin.read(name)
        if name in slide_files:
            xml = data.decode("utf-8")
            # transition must come after cSld, before clrMapOvr
            m = re.search(r"</p:cSld>\s*", xml)
            if m and "<p:transition" not in xml:
                xml = xml[:m.end()] + FADE + "\n" + xml[m.end():]
                patched += 1
            out.writestr(name, xml.encode("utf-8"))
        else:
            out.writestr(name, data)
    out.close()
    zin.close()
    print(f"patched {patched}/{len(slide_files)} slides: {dst}")

if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    main(src, dst)
