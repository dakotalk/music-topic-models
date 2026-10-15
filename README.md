# Topic models of chord progressions in popular music

Latent Dirichlet Allocation over *bargrams* — one bar of harmony, with the
onset and duration of each chord, spelled relative to the local key — on
two corpora:

* the **McGill Billboard** corpus (890 expert-annotated Hot 100 songs, 1958–1991;
  Burgoyne, Wild & Fujinaga 2011), which has metre, section labels and key
  changes, and
* **Chordonomicon** (Kantarelis et al. 2024), 680k crowd-sourced chord sheets
  with section tags, genre and release date but no rhythm or key, of which a
  40,000-song sample with release dates is used.

The question is the one from the original 2015 project: do the "tonal
systems" music theorists describe for rock (Everett 2004; Biamonte 2010;
de Clercq & Temperley 2011) fall out of an unsupervised model, and do the
resulting topics say anything about song structure and chart year?

This is a from-scratch rewrite of that project. The `legacy/` folder keeps the
original notebooks and slides; the section at the end lists what was wrong with
them and what changed.

## Results on Billboard (K = 6)

Six topics, trained on 708 songs and evaluated on 178 held-out songs the model
never saw. Roots are relative to the local tonic (C = I); a token like
`0,2,F:maj---2,2,G:maj` is a bar with IV on beats 1–2 and V on beats 3–4.

| Topic | Share | Top tokens (probability) | Reading | Example songs |
|---|---|---|---|---|
| 0 | 0.38 | I (.49) IV (.28) V (.11) II (.03) | plain diatonic major, one chord a bar | The Joker; Wanna Be Startin' Somethin'; Bust A Move |
| 1 | 0.16 | i (.22) ♭VII (.19) ♭VI (.12) ♭III (.09) iv (.07) i7 (.06) | the flat-side / Aeolian system | Let The Music Play; There's the Girl; Lambada |
| 2 | 0.15 | V (.24) vi (.20) I–IV half-bars (.11) IV–I half-bars (.09) ii (.07) iii (.05) | sharp-side diatonic with faster harmonic rhythm | Just Like Heaven; Here I Go Again; Little Bit O' Soul |
| 4 | 0.13 | V7 (.31) I7 (.23) IV7 (.17) i7 (.08) | the blues: dominant sevenths on I, IV, V | It's Your Thing; Higher Ground; I Got You (I Feel Good) |
| 3 | 0.12 | IV–V half-bars (.19) Imaj7 (.11) IVmaj7 (.10) vi7 (.09) V7 (.08) ii7–V7 (.06) | seventh-chord soul / jazz-pop | Come And Get Your Love; Caught Up In The Rapture |
| 5 | 0.05 | ii7 (.39) then 3/4-time I, IV, V, V7 | waltzes and ii7-heavy ballads | Annie's Song; Could I Have This Dance; Mr. Bojangles |

Full listings with 12 tokens per topic are in [`results/billboard/topics.md`](results/billboard/topics.md).
Topics 0, 1, 2 and 4 are the systems the theory literature predicts: a
"conservative" diatonic style, the flat-side (♭III/♭VI/♭VII) cluster, the
sharp-side (ii/iii/vi) cluster, and blues-based harmony (de Clercq & Temperley's
two main harmonic profiles plus the blues).

### Topics over time

![topic share by period](results/billboard/topic_share_by_period.png)

Mean topic share per chart period, with 95% bootstrap confidence intervals
(151–201 songs per period):

| Topic | 1958–64 | 1965–70 | 1971–76 | 1977–82 | 1983–91 |
|---|---|---|---|---|---|
| 0 diatonic major | 0.40 | 0.40 | 0.40 | 0.39 | 0.35 |
| 1 flat-side minor | **0.07** | 0.16 | 0.14 | 0.17 | **0.26** |
| 2 sharp-side, fast | 0.13 | 0.14 | 0.13 | 0.17 | 0.18 |
| 4 blues sevenths | **0.22** | 0.18 | 0.14 | 0.11 | **0.07** |
| 3 soul sevenths | 0.14 | 0.07 | 0.14 | 0.11 | 0.13 |
| 5 waltz / ii7 | 0.05 | 0.06 | 0.06 | 0.06 | 0.02 |

The two clear trends are the collapse of blues-based harmony (22% → 7%) and the
rise of the flat-side minor system (7% → 26%) over the three decades.

### Downstream tasks, with baselines

Every number is a mean ± sd over the five LDA seeds at K = 6, so model-selection
luck is included in the error bar. Section classification uses GroupKFold so
sections of one song never straddle train and test.

| Task | Features | Score | Baseline |
|---|---|---|---|
| Section type (verse / chorus / bridge / prechorus), 5,551 sections | 6 topic shares | accuracy 0.47 ± 0.03, balanced 0.47 | majority class 0.43 |
| Verse vs chorus, 4,755 sections | 6 topic shares | accuracy 0.61 ± 0.01 | majority 0.49 |
| Chart year, 886 songs | 6 topic shares | MAE 6.1 ± 0.1 years | predict the mean 7.6 |

Harmony tells you a little about song structure and a little about the year —
about 1.5 years better than guessing the mean — and no more than that. The
original slides' claim that bridges and prechoruses were "correctly identified
in most cases" does not survive a correctly labelled confusion matrix:

![confusion](results/billboard/section_confusion.png)

### Choosing K and evaluating coherence

![k sweep](results/billboard/k_sweep.png)

Coherence is mean pairwise NPMI of each topic's top ten tokens, with
co-occurrence counted in **held-out** songs. Measuring PMI on the training
corpus — as the 2015 notebook did, after regularising the model toward high
PMI — rewards the regularisation rather than the topics.

Neither held-out NPMI nor held-out likelihood can choose K here: both keep
rising past K = 20 while the topics degenerate into two or three tokens each
(at K = 32, vocabulary 74, the leading topic is 0.60 IV + 0.29 I + 0.11 V and
nothing else). K is chosen instead by **reproducibility**: train five seeds,
Hungarian-match topics between every pair of runs by cosine similarity, and
count the topics that find a match ≥ 0.7 in *every* other seed. That fraction
peaks at K = 6 (0.50). Seed stability is 0.70 mean matched cosine — far from
perfect, which is why the downstream numbers are averaged over seeds.

### Collocations hurt

The original project's main idea was to glue frequently co-occurring bars into
multi-bar progression tokens (Mikolov-style collocation detection) before
modelling. `scripts/experiment_tokenization.py` tests that on held-out data:

| collocation rounds | doc-freq floor | vocab | held-out NPMI (K=8) | pairs co-occurring in held-out | stability |
|---|---|---|---|---|---|
| 0 | 10 | 76 | **−0.21** | **0.67** | **0.64** |
| 1 | 10 | 108 | −0.26 | 0.63 | 0.52 |
| 2 | 10 | 88 | −0.44 | 0.48 | 0.52 |
| 2 | 3 | 452 | −0.50 | 0.42 | 0.49 |

Every round of collocation detection lowers coherence and stability, and the
2015 configuration (two rounds, floor of 3) is the worst cell. Multi-bar tokens
are near-unique to a handful of songs, so topics built from them never
reproduce on new songs. The full grid (rhythm on/off, rounds, floor, alpha) is
in `results/billboard/tokenizer_grid.csv`; a full run with one collocation
round is in `results/billboard_collocations/` for comparison (its section
classifier does no better than the majority baseline).

## Results on Chordonomicon (40,000 songs)

See [`results/chordonomicon/`](results/chordonomicon/) — `topics.md` with genre
and mode lift per topic, `downstream.txt`, `topic_share_by_decade.png` and
`genre_topic_lift.png`. Chordonomicon has no durations or metre, so each chord
is one unit token; keys are estimated with Krumhansl–Schmuckler profiles
before transposition, which is noisier than Billboard's annotated tonics.

<!-- CHORDONOMICON_SUMMARY -->

## Running it

```bash
pip install -r requirements.txt
python tests/test_parsing.py

# McGill Billboard: annotations + index from https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/
python scripts/run_billboard.py --data data/McGill-Billboard --index data/billboard-2.0-index.csv --out results/billboard
python scripts/run_billboard.py ... --collocations 1 --out results/billboard_collocations
python scripts/experiment_tokenization.py --data ... --index ... --out results/billboard/tokenizer_grid.csv

# Chordonomicon: chordonomicon_v2.csv from https://huggingface.co/datasets/ailsntua/Chordonomicon
python scripts/run_chordonomicon.py --csv data/chordonomicon_v2.csv --out results/chordonomicon --n 40000
```

Runs are deterministic (fixed seeds for the song split, LDA, classifiers and
bootstraps). The Billboard run takes about five minutes on one core; the
Chordonomicon run about thirty.

### Layout

```
chordtopics/
  chords.py         pitch classes, relative roots, chord-quality buckets
  billboard.py      SALAMI annotation parser: metre, x-repeats, dot continuations,
                    per-bar metre overrides, mid-song key changes, section labels
  chordonomicon.py  CSV loader, chord-symbol parser, Krumhansl key estimate
  tokenize.py       bargram tokens, repeat collapsing, collocations, thresholds
  model.py          LDA, held-out NPMI, Hungarian topic matching, seed stability
  evaluate.py       GroupKFold section classifier, year regression, baselines, bootstrap CIs
scripts/            run_billboard.py, run_chordonomicon.py, experiment_tokenization.py
tests/              parser and tokenizer tests
results/            outputs of the runs above (models are not committed)
legacy/             the 2015 notebooks and slides
```

## What changed from the 2015 version

The old notebooks are kept in `legacy/`. Problems found in them, and what this
rewrite does instead:

* **Confusion matrix axes were swapped and mislabelled** — `confusion_matrix(pred, true)`
  with tick labels in a different order from sklearn's. Read correctly, the
  prechorus row was 0/7, not "correctly identified in most cases".
* **No baselines.** 50.7% four-class accuracy was reported without noting that
  "verse" alone is 44% of sections; the year regressor (MAE 7.51) was no better
  than predicting the mean (7.57). Every score here has its baseline beside it.
* **Leakage.** Sections from the same song landed in both train and test;
  now `GroupKFold` by song.
* **Empty documents.** Sections with no surviving tokens were fed to LDA and
  the classifier, each carrying the identical prior as its "features". They
  are now dropped and counted.
* **Circular coherence.** PMI was computed on the training corpus for a model
  regularised toward high PMI. Now NPMI on held-out songs.
* **Whole-song transposition.** Songs were transposed to C by their opening key,
  so a key change turned the rest of the song into "exotic" chords. The parser
  now follows the annotation's `# tonic:` changes (82 of 890 songs).
* **Dropped last bar.** The bargram loop never flushed the final bar of each
  song.
* **Collocations.** See above: they were the headline idea and they hurt.
* **K = 16 by eye**, model trained on section documents but evaluated as if on
  songs, Python 2, hard-coded `C:\` paths, cells run out of order, the MATLAB
  model that did the actual modelling not in the repo. All replaced by a
  package that runs top to bottom with `python scripts/run_billboard.py`.

## References

* Burgoyne, Wild & Fujinaga (2011). An expert ground truth set for audio chord recognition and music analysis. *ISMIR*.
* Kantarelis, Thomas, Lyberatos, Dervakos & Stamou (2024). Chordonomicon: a dataset of 666,000 songs and their chord progressions. arXiv:2410.22046.
* de Clercq & Temperley (2011). A corpus analysis of rock harmony. *Popular Music* 30(1).
* Everett (2004). Making sense of rock's tonal systems. *Music Theory Online* 10(4).
* Biamonte (2010). Triadic modal and pentatonic patterns in rock music. *Music Theory Spectrum* 32(2).
* Newman, Lau, Grieser & Baldwin (2010). Automatic evaluation of topic coherence. *NAACL*.
* Krumhansl & Kessler (1982). Tracing the dynamic changes in perceived tonal organization. *Psychological Review* 89(4).
