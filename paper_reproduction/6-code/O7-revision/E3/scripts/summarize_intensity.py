import json,sys
from pathlib import Path
import numpy as np
out=Path(sys.argv[1]);pga=np.load(out/'inputs_samples.npz')['peak'];g=9.80665
result={'g_ms2':g,'n_samples':len(pga),'strict_exceedance':True,'thresholds':[]}
for c in [.5,1.,1.5]:
    n=int(np.count_nonzero(pga>c*g))
    result['thresholds'].append(dict(threshold_g=c,threshold_ms2=c*g,n_exceeding=n,
                                     fraction=n/len(pga),percent=100*n/len(pga)))
report=json.loads((out/'database_audit.json').read_text());report['input_intensity_coverage']=result
(out/'database_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
(out/'input_intensity_coverage.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
