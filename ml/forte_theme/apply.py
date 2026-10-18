"""Embed local brand assets into hot-served pages; does not restart services."""
from pathlib import Path
import base64,json,re,shutil,time
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PAGES=['training','scenario','analyze','human_review','human_objects','annotator','readiness','live_console/page','training_archive']
def apply():
 logo=ROOT/'frontend/Снимок экрана_20261018_035924.png'
 fonts=HERE/'fonts'
 fontcss=''
 for subset in ['latin','cyrillic']:
  f=fonts/f'inter-{subset}-wght-normal.woff2';data=base64.b64encode(f.read_bytes()).decode()
  urange='U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD' if subset=='latin' else 'U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116'
  fontcss+=f'@font-face{{font-family:Inter;font-style:normal;font-weight:100 900;font-display:swap;src:url(data:font/woff2;base64,{data}) format("woff2");unicode-range:{urange}}}'
 css=fontcss+(HERE/'theme.css').read_text()
 js='window.FORTE_LOGO='+json.dumps('data:image/png;base64,'+base64.b64encode(logo.read_bytes()).decode())+';'+(HERE/'theme.bundle.js').read_text()
 block='<!-- FORTE-THEME-START --><style id="forte-theme">'+css+'</style><script>'+js+'</script><!-- FORTE-THEME-END -->'
 backup=ROOT/'backups'/('forte_design_'+time.strftime('%Y%m%dT%H%M%S'));backup.mkdir(parents=True,exist_ok=True)
 for page in PAGES:
  p=ROOT/'ml'/f'{page}.html';s=p.read_text();shutil.copy2(p,backup/p.name)
  s=re.sub(r'<!-- FORTE-THEME-START -->.*?<!-- FORTE-THEME-END -->','',s,flags=re.S)
  # Insert after existing styles so theme overrides them, before body scripts execute.
  end=s.rfind('</style>')+len('</style>');assert end>=8,page
  pageblock=block
  if page=='analyze':pageblock=block.replace('</style><script>',(HERE/'analysis.css').read_text()+'</style><script>',1)
  s=s[:end]+pageblock+s[end:]
  tmp=p.with_suffix('.html.theme-tmp');tmp.write_text(s);tmp.replace(p)
 print('Updated',len(PAGES),'pages. Backup:',backup)
if __name__=='__main__':apply()
