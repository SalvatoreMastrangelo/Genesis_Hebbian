# Supervisor verdict on chapter 4 (mail of 2026-09-30) and what was done

Andrea's mail (after chapter 3): rather than point edits, make a real cut in the main
text; an appendix can keep the extra parts. In the thesis, and so in chapter 4, focus on
(i) the parts that are new and come from the author's contribution, or (ii) the parts
needed to understand it. Describing the aerodynamic simulation in detail is not needed.
The forest flight environment is the interesting part, since it is used; even there the
mathematics of the collision can be left out.

## Author's decisions (2026-09-30)

1. Survival laws of the blind flight: appendix (words and numbers stay in the main text).
2. Latin hypercube forests: full subsection stays.
3. Appendix: verbatim move of the removed text.

## Applied (chapter 4 and new Appendix A, `THESIS_DECISIONS.md` chapter 4 entry has the map)

- Main text 9 300 → 6 450 prose words; mean sentence 31 → 21 words (chapter-3 rules applied).
- Appendix A "Simulation Details", six sections: Genesis architecture and Taichi kernels;
  rigid body dynamics; aerodynamic model (full, verbatim, neutral wording preserved);
  aerodynamic noise formula; collision test and forest statistics; mass and power models.
- Kept visible in the main text: the credit to the solver's author, the "ranks bodies but
  absolute values unvalidated" caveat, the GPU non-determinism and the "every outcome is
  a random variable" conclusion, the 25 Hz / 100 Hz step figure, all of 4.2 and 4.3.
- Nothing cut outright: the four paragraphs proposed for deletion (list of continuum
  solvers, Taichi compilation pipeline, unused collision pipeline, "young project") are in
  the appendix or, for the last one, still in 4.1.1.

## Still open

- `tab:feasibility` move (chapter 3, rule 7 of the chapter-3 comments): not done.
- Chapter 6 had the writing pass on 2026-09-30 (38 → 21 words per sentence, captions untouched). Chapter 5 has not: it is scheduled for later (author, 2026-09-30). Chapter 2 needs no pass (author, 2026-09-30).
