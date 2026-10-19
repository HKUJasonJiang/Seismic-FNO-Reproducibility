"""Run curated experiments in an isolated, persistent workspace."""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
SCRIPTS = {
 'o1-train': 'O1-baseline-comparison/scripts/train_baselines.py',
 'o1-fno': 'O1-baseline-comparison/scripts/train_fno_large_bs64.py',
 'o1-evaluate': 'O1-baseline-comparison/scripts/evaluate_baselines.py',
 'o2-train': 'O2-data-efficiency/scripts/train_eval_o2.py',
 'o3-pga': 'O3-generalization/scripts/train_eval_pga_ood.py',
 'o3-sampling': 'O3-generalization/scripts/train_eval_sampling_ood.py',
 'o3-classical': 'O3-generalization/scripts/plot_classical_native_output.py',
 'o4-train': 'O4-multi-structure/scripts/train_o4_fno_large.py',
 'o5-export': 'O5-fema-p58/scripts/export_o4_fno_demands.py',
 'o5-frs': 'O5-fema-p58/scripts/compute_floor_response_spectra.py',
 'o5-p58': 'O5-fema-p58/scripts/run_multistructure_p58.py',
 'o5-placement': 'O5-fema-p58/scripts/run_equipment_placement_risk.py',
 'o5-duration': 'O5-fema-p58/scripts/analyze_time_history_duration_metrics.py',
 'e1': 'O7-revision/E1/scripts/benchmark_inference.py',
 'e2-capacity': 'O7-revision/E2/scripts/run_sensitivity.py',
 'e2-boundaries': 'O7-revision/E2/scripts/boundary_sensitivity.py',
 'e3': 'O7-revision/E3/scripts/audit_database.py',
 'e3-intensity': 'O7-revision/E3/scripts/summarize_intensity.py',
 'e4': 'O7-revision/E4/scripts/rope_resolution.py',
 'e5-cache': 'O7-revision/E5/scripts/cache_data.py',
 'e5': 'O7-revision/E5/scripts/train_candidate.py',
 'e5-additional': 'O7-revision/E5_additional_20261017/scripts/train_candidate.py',
}

def prepare(workspace, data_root):
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PACKAGE/'6-code', workspace/'6-code', dirs_exist_ok=True)
    # Seed ONLY postprocessing inputs. Never seed archived training results into a fresh run.
    for name in ['O1-baseline-comparison','O2-data-efficiency','O3-generalization','O4-multi-structure','O5-fema-p58']:
        src=PACKAGE/'recorded/6-code'/name
        if src.exists():
            for f in src.rglob('*'):
                if f.is_file() and 'runs' not in f.relative_to(src).parts:
                    dst=workspace/'6-code'/name/f.relative_to(src)
                    if not dst.exists(): dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
    cfgpath=workspace/'6-code/O5-fema-p58/configs/o5_multistructure_cases.json'
    cfg=json.loads(cfgpath.read_text());cfg['base_data_dir']=str(data_root)
    cfg['gm_local_path']=str(data_root/cfg['gm_hf_path'])
    index={'structures':{}}
    for sid,v in cfg['structures'].items():
        v['local_hdf5_path']=str(data_root/v['hdf5_hf_path'])
        index['structures'][sid]={'checkpoint_path':str(workspace/'6-code/O4-multi-structure/results/runs'/sid/'checkpoints'/f'FNO-Large_{sid}_best.pt')}
    cfgpath.write_text(json.dumps(cfg,indent=2))
    p=workspace/'6-code/O5-fema-p58/input/o4_artifacts/artifact_index.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(index,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=PACKAGE/'runs/workspace')
    p.add_argument('--data-root',type=Path,default=PACKAGE/'data')
    p.add_argument('experiment',choices=['prepare',*SCRIPTS])
    p.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args();work=a.workspace.resolve();data=a.data_root.resolve()
    if work==PACKAGE or work==REPO or PACKAGE/'recorded' in [work,*work.parents]:
        p.error('Choose a separate run workspace, not the repository or recorded evidence.')
    prepare(work,data)
    if a.experiment=='prepare':print(work);return
    args=a.args[1:] if a.args[:1]==['--'] else a.args
    if a.experiment in ['e5','e5-additional']:
        if '--root' in args:p.error('--root is managed by this runner')
        args=['--root',str(work),*args]
    if a.experiment in ['o1-train','o1-fno','o1-evaluate','o2-train','o3-pga','o3-sampling','o4-train']:
        if '--base_data_dir' not in args:args+=['--base_data_dir',str(data)]
        if a.experiment=='o4-train':
            if '--data_dir' not in args:args+=['--data_dir',str(data/'MDOF/knet-250/Data/fno')]
        elif '--building_dir' not in args:args+=['--building_dir',str(data/'f6_only')]
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(REPO),str(REPO/'module'),str(work/'6-code/shared/utils'),str(work/'6-code/O1-baseline-comparison/scripts'),env.get('PYTHONPATH','')])
    env.setdefault('MPLBACKEND','Agg')
    env['PYTHONDONTWRITEBYTECODE']='1'
    raise SystemExit(subprocess.call([sys.executable,str(work/'6-code'/SCRIPTS[a.experiment]),*args],cwd=work,env=env))

if __name__=='__main__':main()
