"""Read GNU time -v records produced for each native process."""
import re
from pathlib import Path

def process_metrics(path):
    text=Path(path).read_text()
    elapsed=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([\d:.]+)',text)
    rss=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',text)
    if not elapsed or not rss:
        raise ValueError('Missing native process resource metrics: '+str(path))
    seconds=0.0
    for part in elapsed.group(1).split(':'):
        seconds=seconds*60+float(part)
    return {'seconds':seconds,'peak_rss_kib':int(rss.group(1))}
