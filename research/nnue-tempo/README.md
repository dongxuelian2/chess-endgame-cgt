# The value of having the move in Stockfish NNUE

[Integrated endgame study (PDF)](manuscript/paper.pdf) · [Evidence and reproduction guide](manuscript/README.md)

[Research paper (PDF)](paper.pdf) · [Manuscript source](paper.typ) · [Annotated primary literature](literature.md) · [Numerical results](results.json) · [Four-man census and 48 chess tests](atlas/README.md)

The experiment compares static evaluation on the same placement with either side to move. Syzygy supplies the paired outcome in fourteen material classes. Stockfish 12, 14.1 and 18 cover three generations of NNUE. The research includes a full KPK census, layer-wise interventions, a replication on played positions, and controlled move selection. The classical and neural static evaluations come directly from each release that includes both. Stockfish 18 supplies a separate large-network input-channel intervention.

## Reproduction

Requirements are a C++ compiler, GNU Make, Git, curl, zstd, Python with NumPy and Matplotlib, and Typst for PDF generation. The recorded runs used AVX2, Python 3.14, NumPy 2.5.4, and Matplotlib 3.11.2. Run from the repository root in Bash. The setup script takes a fresh directory and verifies network digest prefixes. It downloads the [Lichess rated-games corpus](https://database.lichess.org/), eight additional Syzygy WDL tables, and the pinned python-chess checkout with its three- and four-man tables. The evaluator and search runs execute on CPU.

```bash
study=research/nnue-tempo
scratch=.scratch/nnue-reproduction
bash "$study/setup.sh" "$scratch"
export PYTHONPATH="$PWD/$scratch/pychess"
python "$study/collect.py" "$scratch" --workers 6
python "$study/collect.py" "$scratch" --extended --count 32000 --workers 6
python "$study/analyze.py" "$scratch" > "$scratch/results.json"
python "$study/analyze.py" "$scratch" --extended > "$scratch/results5.json"
python "$study/opposition.py" "$scratch"
python "$study/census18.py" "$scratch" > "$scratch/census18.json"
python "$study/game_pairs.py" "$scratch"
python "$study/modern.py" "$scratch" > "$scratch/modern.json"
python "$study/move_test.py" "$scratch"
python "$study/pawn_edit.py" "$scratch" > "$scratch/pawn-edits.json"
python "$study/surrogate.py" "$scratch" > "$scratch/surrogate.json"
python "$study/replicate_unit.py" "$scratch" > "$scratch/replication.json"
python "$study/verify_native.py" "$scratch"
python "$study/summarize.py" "$scratch" "$study"
typst compile "$study/paper.typ" "$study/paper.pdf"
```

The generator stores one row per placement with both side-to-move scores, labels, king-configuration group, immediate move indicators, and 14.1 dense activations. Full rows and move choices remain in the scratch directory. The public JSON records family-level AUCs, confidence intervals, all single-unit ablations, exact KPK confusion matrices, and the paired move totals. Column order in each move-count array is classical successes, default successes, default-only successes, classical-only successes. The KPK matrix rows encode release, evaluator (0 classical, 1 NNUE), sum of Syzygy labels, predicted sign, and count. The SF18 matrix omits the first two fields. The pawn edits are reconstructed directly from the held-out placements by the corresponding script.

## Verification

The analysis reconstructs both dense transforms downstream of the 14.1 eight-unit layer and checks the final integer score for both turns of every sampled placement. Native UCI validation compares the recorded scores to independently built clean binaries. The SF18 knockout requires exact equality for placements with zero active FullThreats inputs. Every root in the move test additionally satisfies the Syzygy Bellman equality over all legal successors.
