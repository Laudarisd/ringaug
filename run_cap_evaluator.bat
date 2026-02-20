@echo off
setlocal

REM -------- Settings (edit these) --------
set ORIG_DIR=.\seg-topo-augment\json
set ROBOFLOW_DIR=.\roboflow\json
set INDEXED_DIR=.\seg-topo-augment\augmented\augmented_index_json
set DIST_THRESH=20
REM ---------------------------------------

echo ========================================
echo Running RoboFlow CAP Evaluation...
echo Distance threshold: %DIST_THRESH%
python cap_evaluator.py --mode roboflow --orig-dir "%ORIG_DIR%" --roboflow-dir "%ROBOFLOW_DIR%" --all --dist-thresh %DIST_THRESH%

echo ========================================
echo Running SegTOPO Indexed CAP Evaluation...
python cap_evaluator.py --mode segtopo --indexed-dir "%INDEXED_DIR%" --all

echo ========================================
echo RoboFlow results: roboflow\cap_evaluation\final_results.txt
echo SegTOPO results: seg-topo-augment\cap_evaluation\final_results.txt

endlocal
