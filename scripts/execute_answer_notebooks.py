"""Execute only filled-in instructor notebooks using this project interpreter."""

from pathlib import Path
import json
import os
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/causal_study_20261015"


def main():
    destination=OUT/"executed"
    runtime=ROOT/"tmp/notebook-kernels"
    destination.mkdir(parents=True,exist_ok=True)
    runtime.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in ["workshop_answers","causal_cafe_complete","causal_forest_appendix"]:
        source=ROOT/"solutions"/f"{name}.ipynb"
        notebook=nbformat.read(source,as_version=4)
        km=KernelManager(kernel_name="python3",connection_file=str(runtime/f"{name}.json"))
        km.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
        def progress(cell,cell_index,**kwargs):
            if cell.cell_type=="code":
                print(f"  code cell {cell_index+1}",flush=True)
        client=NotebookClient(notebook,km=km,timeout=300,startup_timeout=30,
                              on_cell_start=progress,resources={"metadata":{"path":str(ROOT)}})
        env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
             "PYTHONUTF8":"1","CAUSAL_CAFE_OUTPUT_DIR":str(OUT/"classroom_analysis")}
        started=time.perf_counter()
        print("Executing:",name,flush=True)
        try:
            client.execute(cwd=str(ROOT),env=env)
            nbformat.validate(notebook)
            nbformat.write(notebook,destination/f"{name}.executed.ipynb")
            record={"source":source.relative_to(ROOT).as_posix(),"status":"passed",
                    "elapsed_seconds":time.perf_counter()-started,
                    "code_cells":sum(c.cell_type=="code" for c in notebook.cells),
                    "outputs":sum(len(c.get("outputs",[])) for c in notebook.cells)}
            records.append(record)
            print(record,flush=True)
        except BaseException:
            nbformat.write(notebook,destination/f"{name}.failed.ipynb")
            raise
        finally:
            if km.has_kernel:
                km.shutdown_kernel(now=True)
            if client.kc is not None:
                client.kc.stop_channels()
    (OUT/"notebook_execution.json").write_text(json.dumps(records,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
