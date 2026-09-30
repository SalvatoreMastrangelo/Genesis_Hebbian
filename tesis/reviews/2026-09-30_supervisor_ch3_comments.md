# Supervisor comments on chapter 3 (annotated PDF of 2026-09-30)

Source: `~/Downloads/thesis.pdf`, 20 annotations (18 highlights, 2 ink marks), all on
thesis pages 28–30 (PDF pages 29–31), i.e. the intro of chapter 3, §3.1, §3.2 and the
table of §3.3. Nothing after Table 3.1 is annotated.

**Applied 2026-09-30 to the whole of chapter 3** (author's command: "apply the rules to chapter 3 only"), rules 1–6 and 8 in every paragraph, annotated or not. Rule 7 (move Table 3.1 to the results) NOT applied: it touches chapters 4, 5 and 6, destination to be decided (§3 below).

Cover mail (Andrea): there is quite some work to do on the writing; many sentences are
too long and convoluted, which makes it hard to understand what is meant; the comments
are examples, extract their essence and apply it to the rest of the thesis. He is going
on with chapter 4.

TeX file: `tesis/chapters/03_Challenges/03_Challenges.tex` (one paragraph per line):
intro = L4, L6; §3.1 = L11, L13, L15; §3.2 = L22, L29, L31; §3.3 = L38, table L40–56.

## 1. The rules behind the comments (to apply to every chapter)

1. **One idea per sentence.** Split long sentences. Fewer parenthetical insertions
   (incisi) and subordinate clauses; instead put a full stop and start again.
   His words on ch3:13 "A task that rewards both, …": "questo tipo di frasi è molto
   difficile da leggere … piuttosto, metti un punto e riparti con una nuova frase.
   aiuta molto nella scrittura scientifica". Same on ch3:29 ("troppo lunga") and
   ch3:22 ("qui ci va un ." because the second half of the sentence talks about
   something else).
2. **Name the referent every time, even at the cost of repeating it.** "either" and
   "both" in "Other points may be better on either objective, or dominate on both"
   have no visible antecedent. Rule: no bare "it / them / both / either / the two /
   the same / one level up" when the noun is more than a clause away.
3. **Say things in the positive.** The roadmap paragraph ("the first shows what is
   lost when …, the second what is lost when …") should say what each challenge IS
   and what its impact on the goals of the thesis is, not what is lost.
4. **Plain verbs, no loaded or idiomatic words.** "judged" → "evaluated" (he would
   avoid "judged" altogether); "how far it gets" → "goes / travels"; "in flight" →
   "during flight"; "ranked" not understood; "at all" and "one level up" marked "?".
   Rule: when a word is a metaphor or an idiom, replace it with the literal word.
5. **Make the causal link explicit, then give the example.** On ch3:13 he rewrites
   the paragraph opening as "because the body design constrains what the controller
   can do. For example, energy per metre …": state the claim, then "For example",
   then the instances. The same pattern is behind "With respect to (2.27), this
   corresponds to fixing m …" (ch3:11) and "Usually, flying robot design starts from
   the body. …" (ch3:11): lead with the plain statement, then the formal one.
6. **The key sentence of a section must be precise and visible.** The closing
   sentence of §3.1 ("The sequential challenge is therefore the reason to search over
   bodies at all: whatever the controller, a body that was never tried cannot be
   found") is not understood; find a more precise formulation of the challenge and,
   since so much weight is put on it, set it in `\emph`. Apply to the closing
   sentence of §3.2 and §3.3 as well, and to every "therefore" sentence of chapters
   4–6.
7. **Keep the level of detail of the chapter.** Table 3.1 is "too technical" for the
   challenges chapter: the discussion of computational efficiency (numbers, GPUs,
   hours) can go with the experimental results; chapter 3 keeps a generic discussion
   of the challenge. This matches the author's own rule in `THESIS_DECISIONS.md`
   ("no platform details in chapter 3").
8. **Transitions in one plain sentence.** The opening of §3.3 ("The remedy that the
   previous section seems to call for is the one adopted by …") is replaced by
   "The previous sections suggest that brain–body co-design may represent a promising
   direction for our thesis."

## 2. Comment by comment

| # | Where | Highlighted text | Comment (translated) |
|---|---|---|---|
| 1 | intro, L4 | "judged on how far it gets" | "evaluated"; avoid "judged" as a term. "gets" → "goes? travels?" |
| 2 | intro, L4 | "a performance F(m, θ) that can only be measured by flying them" | → "which evaluates the quality of the flight"; "flying them" seems to refer to the controller too, which is odd |
| 3 | intro, L6 | whole roadmap paragraph | not clear; rephrase in the positive instead of "what is lost"; highlight the challenges and their impact on the goals of the thesis |
| 4 | §3.1, L11 | "The usual way of building a flying robot is sequential." | suggested: "Usually, flying robots design starts from the body. etc etc" |
| 5 | §3.1, L11 | "In the terms of (2.27)" | suggested: "With respect to (2.27), this corresponds to fixing m etc etc" (he wrote 2.21; the equation is `eq:codesign-bilevel`) |
| 6 | §3.1, L13 | "Energy per metre …" | suggested lead-in: "because the body design constrains what the controller can do. For example, energy per …" |
| 7 | §3.1, L13 | "can change in flight" | "during" |
| 8 | §3.1, L13 | "A task that rewards both, distance through a cluttered forest and efficiency along the way, asks for a trade-off between the two, and a single hand-designed body sits at one point of that trade-off, chosen before the task was specified." | very hard to read; simpler structure, fewer incisi and subordinates; full stop and new sentence |
| 9 | §3.1, L13 | "Other points may be better on either objective, or dominate on both." | always say what you refer to, even at the cost of repeating; "either" and "both" are unclear |
| 10 | §3.1, L15 | "at all:" | "?" |
| 11 | §3.1, L15 (ink) | whole closing sentence | find a more precise formulation of the challenge; since it is given much weight, put it in `\emph` |
| 12 | §3.2, L22 | "fix the controller and search over the body" | "the space of possible bodies" |
| 13 | §3.2, L22 | "… any body in the search space, scores every candidate, …" | a full stop goes here: what follows talks about something else |
| 14 | §3.2, L22 | "Two bodies can be ranked one way under the shared controller and the other way under their own" | "ranked" not understood |
| 15 | §3.2, L29 | "A generalist controller is trained on a distribution of bodies and is, on any one of them, a compromise between all of them." | not clear, rephrase |
| 16 | §3.2, L29 | "But the edges are where the search wants to go: a body that is better than the reference is, almost by definition, one that departs from it, in size, in shape, in control authority, and the more it departs the more it is penalized by a controller that was never adapted to it." | too long |
| 17 | §3.2, L31 | "The same argument applies one level up." | "?" |
| 18 | §3.3, L38 | "The remedy that the previous section seems to call for is the one adopted by the co-design works that train their controllers by reinforcement learning" | suggested: "The previous sections suggest that brain–body co-design may represent a promising direction for our thesis." |
| 19 | §3.3, Table 3.1 (ink) | the table | too technical; the computational-efficiency discussion could go with the experimental results; here keep the discussion generic on the challenges |

(#10 and #11 are two annotations on the same sentence.)

## 3. What moving Table 3.1 touches (decision for the author)

- `\autoref{tab:feasibility}` is pointed to from 4.1.2 and from 5.1 (kept on purpose,
  `THESIS_DECISIONS.md`, pointers entry of 2026-09-21). If the table moves, the two
  pointers follow it, or 5.1 restates the two measured times.
- The paragraph after the table (ch3:58, "A single training of the generalist takes
  fourteen hours … a factor of almost fifteen") reads the numbers off the table and
  goes with it. The next paragraph (ch3:61, the noise multiplier: sixty forests per
  body, the score is worthless once the controller moves) and the closing paragraph
  (the shape any answer must have) are the generic argument and stay.
- Candidate destinations: the settings block at the beginning of chapter 6 (the
  author's decision A5 of 2026-09-21 puts every experimental parameter there) or a
  short cost subsection of chapter 6. The bold "(Proposed Co-Design Pipeline)" row
  becomes natural there, since chapter 6 may name the method.
- The `tab:feasibility` A1 leftover ("sixty forests", ch3:61) is still open.

## 4. Where the same rules bite in the other chapters (rough scan, no edits)

Sentence lengths from the TeX sources, LaTeX stripped (the splitter is crude, treat
the numbers as an order of magnitude). Chapters 1 and 7 are not written yet.

| chapter | sentences | mean words | over 40 words |
|---|---|---|---|
| 2 | 192 | 39 | 83 (43 %) |
| 3 | 50 | 28 | 8 (16 %) |
| 4 | 298 | 31 | 89 (30 %) |
| 5 | 323 | 29 | 80 (25 %) |
| 6 | 123 | 38 | 57 (46 %) |

Chapter 3, the one he read, has the shortest sentences of the thesis. Chapters 2 and 6
average ten words more per sentence, so the same criticism will land harder there.
The longest sentences found: ch. 2 LSTM definition (123 words, §2.2), ch. 4 centre of
pressure sentence (109 words, §4.1.4), ch. 2 actor-critic TD update (106 words),
ch. 2 Markov property (101 words), ch. 2 weight clipping vs weight decay (95 words),
ch. 6 "The body that flies farthest is small and light …" (90 words).

Flagged words elsewhere: "judged / judge(s)" 9× in ch. 5, 2× in ch. 2, 2× in ch. 4,
1× in ch. 6; "at all" 5× in ch. 6, 3× in ch. 5, 2× in ch. 4; "rank / ranked /
ranking" throughout ch. 5 and 6 (there it is a technical term of NSGA-II and CMA-ES,
so define it once and keep it, but not as a plain verb).

## 5. Open questions for the author

1. Where does Table 3.1 go (chapter 6 settings block, a cost subsection, or an
   appendix)? Does ch3:58 go with it in full?
2. "ranked" in §3.2: replace with "ordered" / "the search prefers m1 to m2", or keep
   and define?
3. Does the `\emph` rule extend to the closing sentences of §3.2 and §3.3?
4. Should the writing pass on chapters 2, 4, 5, 6 wait for his comments on chapter 4,
   or start now on chapter 3 alone so that he sees the new style on chapter 4?
   ANSWERED 2026-09-30: chapters 3, 4 and 6 done the same day; chapter 5 later; chapter 2 exempt.
