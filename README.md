# AQUALOC code and figures across all revisions

This archive contains the code and figures selected from the requested AQUALOC_Complete_Findings_Revision5 folder. Original relative paths and included file bytes are preserved. There are 239 source files and figure assets: 142 code/configuration source files, 84 figure files, and 13 additional configuration/build records.

The revision/ folder includes the frozen pairwise engine, detector/preprocessing experiments, geometry controls, chronological and native-pose evaluators, Sequence 05 transfer and order-repeat code, ORB-SLAM3 adapters/evaluators, validation scripts, and trajectory/costmap graph generation. Earlier implementation code remains under original_numeric_evidence/. source/ retains document construction and equation helpers that do not embed paper prose. baseline_tools/ preserves adapter/build source and build records. Figures from the original package and Revisions 1–5 remain in their original figures/ folders, including the latest standalone and manuscript-sized graph variants.

Manuscripts, supplements, derivations, Word files, prose Markdown sources, and findings/status documents are excluded. Fourteen Python authoring/packaging scripts that embed manuscript or findings prose are also excluded, so paper text is not carried inside code. Remaining construction helpers may refer to these excluded resources; this is not a document-building package.

Numerical result tables, trajectories, raw images, correspondence caches, serialized checkpoints, transcripts, installers, and compiled binaries are excluded. Figure PDFs are retained only inside figures/ directories; manuscript/supplement PDFs are absent. Repeated historical figure assets are retained deliberately to preserve all revisions.

This is a source-and-figure archive, not a self-contained dataset or full reproduction environment. Scientific scripts require separately supplied AQUALOC images/numerical inputs and their Python/native dependencies. Original scripts retain their historical dataset/output paths; configure AQUALOC_ROOT and adjust remaining explicit path constants when relocating. New trajectory/error-map graph code requires the corresponding saved evaluation inputs. No scientific experiment was rerun while packaging.

MANIFEST.csv records the byte size and SHA-256 of every payload except itself. CRC, archive membership, and every payload hash were verified after packing. Nothing was pushed to GitHub.
