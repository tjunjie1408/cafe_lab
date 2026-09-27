# Causal ML with Xiaohei: deck plan

Built with `ian-xiaohei-illustrations` in talk mode (`references/talk-mode.md`).

## 1. Frame

- **Audience:** students or a general technical audience who know what "prediction" means, but not causal inference.
- **Length:** 11 slides, about 10-12 minutes.
- **The one sentence:** *Prediction asks "what will grow?"; causal ML asks "what changes if we act?". They need different evidence.*
- **Format:** 16:9 slides. The image is full-bleed; the title is typeset on the slide, not in the image.

## 2. Story arc

| Beat | Slides |
|---|---|
| Hook | 1 |
| Familiar ground: prediction | 2-3 |
| The turn: acting, not watching | 4 |
| The trap: confounding + counterfactuals | 5-6 |
| Tools: randomize, measure, adjust | 7-9 |
| Where ML fits, and its limits | 10 |
| Callback | 11 |

## 3. World bible

- **Setting:** a small garden bench with potted plants on pure white paper. There is no ground scenery beyond pots and a thin soil line.
- **Xiaohei:** a tall, rounded solid-black bean body, two small white dot eyes near the top, thin stick legs, thin arms only when holding something. Deadpan and serious. Use the style of the existing "What will grow? / What changes?" image as the first reference.
- **Recurring props (fixed meanings):**
  - **Tap / watering can = the intervention** (watering)
  - **Ruler = measuring the outcome** (height)
  - **Sun = the hidden common cause** (confounder)
  - **Clipboard of past plants = observed data**
  - **Coin = random assignment**
  - **Dashed pencil outline = something not observed** (a prediction, or the other world)
- **Colors:** orange = the main flow / the watering action. Red = the key problem or result. Blue = water drops, predictions, secondary notes.
- **Label words (use exactly these, and no others):** why? · past plants · new plant · guess · more water? · sun · taller · mixed up! · other world · heads · tails · difference · give or take · sunny · shady · assumption · each kind · What will grow? · What changes?

## 4. Beat sheet

| # | Beat | Slide title (typeset) | Image: what Xiaohei does | Labels |
|---|---|---|---|---|
| 1 | Hook | Two plants. Why is one taller? | holds a ruler against the taller of two pots, staring | why? |
| 2 | Prediction | Prediction: learn from past plants, guess the next one | reads a clipboard of past plants, draws a dashed guess over a seedling | past plants · new plant · guess |
| 3 | Checking | A good prediction is checked on plants it has never seen | lifts a curtain to compare dashed guesses with the real plants | guess |
| 4 | Turn | New question: what happens if we water more? | hand on the tap, hesitating, thinking of a taller plant | more water? |
| 5 | Trap | The trap: sunny plants get more water *and* grow taller | waters the sunny side a lot; a big sun links both | sun · more water? · taller · mixed up! |
| 6 | Counterfactual | We never see the same plant both ways | peeks through a fold in the paper at the dashed "other world" version of the pot | other world |
| 7 | Randomize | Fix: let a coin decide who gets water | flips a big coin; heads pots get water, tails pots don't | heads · tails |
| 8 | Effect | The effect is an average difference, with uncertainty | on a ladder with a long ruler across the two groups | difference · give or take |
| 9 | Adjust | No coin? Compare like with like, and say what you assume | sorts pots onto sunny and shady shelves, ties an assumption tag | sunny · shady · assumption |
| 10 | ML | ML handles many measured details and finds who benefits most | cranks a strange machine that turns plant cards into ruler cards of different lengths | each kind |
| 11 | Callback | Same data, same tools. A different question. | sits between the two plants from slide 1; two signposts | What will grow? · What changes? |

## 5. Prompts

Each prompt is **base prompt + continuity block + slide block**. Paste all three into one message. From slide 2 on, attach the approved slide-1 image as the anchor.

### Base prompt (same for every slide)

```text
Generate one standalone 16:9 horizontal English presentation illustration.

Visual DNA:
Pure white background. Minimalist black hand-drawn line art. Slightly wobbly pen lines. Lots of empty white space. Sparse red/orange/blue handwritten English annotations in casual pen lettering. Clean absurd product-sketch feeling. No gradients, no shadows, no paper texture, no complex background, no commercial vector style, no PPT infographic look, no cute mascot poster, no children's illustration, no realistic UI.

Recurring IP character required:
Xiaohei, a small solid-black absurd creature with white dot eyes, tiny thin legs, blank serious expression, slightly uneven hand-drawn body shape. Xiaohei must perform the core conceptual action, not decorate the scene. Make Xiaohei serious, deadpan, and slightly bizarre, not cute. Do not write the name "Xiaohei" on the image.

Color use:
Black for main line art and Xiaohei. Orange for the watering action and main flow arrows. Red only for the key problem or result. Blue only for water drops, predictions, and secondary notes.

Constraints:
One image explains only one idea. Keep the main subject around 40%-60% of the canvas. Preserve at least 35% blank white space. Use only the labels listed below, spelled exactly as written, in small handwritten lettering; no other text. No title, no slide number. It should be clear but not instructional, interesting but not childish, strange but clean.
```

### Continuity block (fill in n)

```text
Series continuity (slide {n} of 11):
Same world and drawing style as the attached anchor image: a small garden of potted plants on pure white paper, thin soil lines, no scenery.
Xiaohei looks identical to the anchor: a tall rounded solid-black bean body, two small white dot eyes near the top, thin stick legs, thin arms only when holding something, same size relative to the pots.
Recurring props keep their look and meaning: tap or watering can = watering / ruler = measuring height / sun = the hidden common cause / clipboard = past plant records / coin = random choice / dashed pencil outline = something not observed.
Same line weight, same pure white background, same handwritten label lettering, same color meanings.
```

### Slide blocks

**Slide 1: Hook** (generate first; this becomes the anchor. Attach the existing "What will grow?" image as a style reference only.)
```text
Theme: two potted plants, one clearly taller.
Structure type: Concept metaphor, close-up.
Core idea: a visible puzzle - why is one plant taller than the other?
Composition: two identical terracotta-style pots drawn in line art side by side, left plant tall, right plant short. Xiaohei stands between them, holding a long wooden ruler upright against the tall plant, head tilted slightly, staring blankly at the top leaf. Large white space on the right third.
Suggested elements: two pots / ruler / a few soil crumbs.
Labels: "why?" in red, small, near Xiaohei's head.
```

**Slide 2: Prediction**
```text
Theme: learning from past plants to guess a new one.
Structure type: Workflow.
Core idea: prediction uses records of past plants to guess the height of a new plant.
Composition: left - Xiaohei holds a clipboard showing three tiny plant sketches with a sun symbol and a water drop next to each. An orange arrow goes right to a small seedling in a pot. Above the seedling, a blue dashed pencil outline shows the guessed full-grown height.
Suggested elements: clipboard / three tiny plant records / seedling / dashed outline.
Labels: "past plants" (black, near clipboard) / "new plant" (black, near pot) / "guess" (blue, near dashed outline).
```

**Slide 3: Checking on unseen plants**
```text
Theme: testing predictions on plants the model has not seen.
Structure type: Before / after, one wide row.
Core idea: a prediction earns trust when its guesses match plants that were hidden from it.
Composition: a row of four pots behind a simple hanging cloth. Xiaohei pulls the cloth aside with one arm. Each pot has a blue dashed guess outline; three of them match the real plants closely, one is a bit off.
Suggested elements: hanging cloth on a thin line / four pots / dashed outlines.
Labels: "guess" (blue, once, near one dashed outline).
```

**Slide 4: The turn**
```text
Theme: the question changes from watching to acting.
Structure type: Character states, single close-up.
Core idea: now we want to know what happens if we do something - water more.
Composition: a garden tap on a thin pipe above one potted plant. Xiaohei grips the tap handle, frozen mid-turn. A small orange curved arrow shows the turning direction. A blue thought bubble above Xiaohei holds a taller version of the same plant, drawn dashed.
Suggested elements: tap / one pot / thought bubble / dashed taller plant.
Labels: "more water?" (orange, near the tap).
```

**Slide 5: The trap (confounding)**
```text
Theme: sunlight drives both watering and growth.
Structure type: System close-up, wide.
Core idea: sunny plants dry out, so they get watered more, and sun also makes them grow taller - so watering and height are mixed up with sun.
Composition: a big sun top-left. Under it, three tall plants; Xiaohei empties a large watering can over them with many blue drops. On the right, in a shaded area drawn with light hatching, three short plants with dry soil. From the sun, two thin black lines: one to the watering can, one to the tall plants.
Suggested elements: sun / watering can / tall plants / shaded short plants / two lines from the sun.
Labels: "sun" (black, by the sun) / "more water?" (orange, by the can) / "taller" (black, by the tall plants) / "mixed up!" (red, between the two lines).
```

**Slide 6: The other world (counterfactual)**
```text
Theme: we can only see one version of each plant.
Structure type: Before / after split.
Core idea: the same plant cannot be both watered and unwatered; the other version is never observed.
Composition: the page seems folded down the middle with a slight fold line. Left: a solid-drawn watered plant in a pot with blue drops. Right: the same pot and plant drawn only as a faint dashed pencil outline, no water. Xiaohei stands at the fold, lifting a corner of the page and peeking at the dashed side.
Suggested elements: fold line / solid plant / dashed twin / lifted page corner.
Labels: "other world" (blue, above the dashed side).
```

**Slide 7: Randomize**
```text
Theme: a coin decides which plants get water.
Structure type: Workflow.
Core idea: random choice makes the two groups alike in sun and everything else, on average.
Composition: centre - Xiaohei flips a large coin that spins in the air above it. Left group: three seedlings getting water from a tap with an orange arrow. Right group: three seedlings, dry. A small, mild sun at the top centre shines equally on both groups.
Suggested elements: big coin / two groups of three seedlings / tap / small sun.
Labels: "heads" (orange, over the watered group) / "tails" (black, over the dry group).
```

**Slide 8: Measure the effect**
```text
Theme: the average difference between the two groups.
Structure type: Concept metaphor, close-up.
Core idea: the effect is the difference in average height, and it comes with uncertainty.
Composition: the two groups from slide 7, now grown; watered group slightly taller on average. Xiaohei stands on a small stepladder holding a long ruler horizontally across the tops. A red double-headed arrow between the two average heights. Around the arrow's ends, a short light-blue band shows uncertainty.
Suggested elements: stepladder / long ruler / two plant groups / red arrow / blue band.
Labels: "difference" (red, by the arrow) / "give or take" (blue, by the band).
```

**Slide 9: No coin? Adjust**
```text
Theme: compare plants with similar sunlight.
Structure type: Method layers.
Core idea: without a coin, compare watered and unwatered plants within the same sunlight level - and state the assumption that sunlight is the thing that matters.
Composition: a simple two-shelf rack. Top shelf with a small sun symbol: two watered pots and two dry pots. Bottom shelf with a small cloud symbol: the same. Xiaohei carries one pot toward the correct shelf. A paper tag on a string hangs from the rack.
Suggested elements: two-shelf rack / eight pots / sun and cloud symbols / tag on string.
Labels: "sunny" (black, top shelf) / "shady" (black, bottom shelf) / "assumption" (red, on the tag).
```

**Slide 10: Where ML fits**
```text
Theme: a machine that estimates the effect for each kind of plant.
Structure type: Concept metaphor.
Core idea: machine learning helps adjust for many measured details and estimates how the watering effect differs across kinds of plants; it still relies on the same assumptions.
Composition: a strange hand-cranked box machine. Plant record cards go in a top slot. Out of a side tray come three small ruler cards of different lengths, each with a tiny different leaf shape. Xiaohei turns the crank seriously. The assumption tag from slide 9 still hangs on the machine.
Suggested elements: crank machine / input cards / three ruler cards / assumption tag.
Labels: "each kind" (orange, by the output tray) / "assumption" (red, on the tag).
```

**Slide 11: Callback**
```text
Theme: two questions, two signposts.
Structure type: Map route, minimal.
Core idea: the same garden supports two questions - what will grow, and what changes if we act.
Composition: the two plants from slide 1, tall and short. Xiaohei sits calmly between them, the ruler across its lap. A signpost with two arrows: the left arrow points to a clipboard on the ground, the right arrow points to a tap. A small sun at the top right.
Suggested elements: two plants / signpost / clipboard / tap / ruler.
Labels: "What will grow?" (blue, left arrow) / "What changes?" (orange, right arrow).
```

## 6. Speaker notes

1. **Hook.** Here are two plants from the same garden. One is clearly taller. The natural question is: why? And the follow-up: if I wanted my short plant to grow, what should I do? Those sound like the same question. By the end of this talk, you'll see they're not.
2. **Prediction.** The first thing machine learning is great at is prediction. Take records of past plants (how much sun, how much water, how tall they got), learn the pattern, and guess the height of a new plant. No need to know why. We only need the pattern to hold.
3. **Checking.** How do we know a prediction is good? We hide some plants, guess, then reveal them. If the guesses match plants the model has never seen, we trust it. This is held-out accuracy, and it's the gold standard for prediction.
4. **The turn.** Now a different question. Not "how tall will it be?" but "if I turn this tap and water more, will it grow taller?" That's an intervention: we act instead of watching. Prediction answered the first question. Does it answer this one?
5. **The trap.** Here's the catch. In this garden, the sunny side dries out, so it gets watered more. Sun also makes plants grow. In the records, watered plants are taller, but part of that is just the sun. Watering and sunlight are mixed up. A model can predict height perfectly well and still give us the wrong answer about watering.
6. **The other world.** The deeper problem: for any plant, we only ever see one version. If it was watered, we never see how it would have grown without water. That other world is missing. So a causal effect is always a comparison with something we can't observe directly.
7. **Randomize.** The cleanest fix is a coin. Let the coin decide which plants get water. Now sunny and shady plants end up in both groups, roughly evenly. Sun can't sneak into the comparison, because the coin doesn't know about the sun.
8. **Effect.** Then we measure. The effect of watering is the difference in average height between the two groups. It's an average, not a promise for every single plant. And because we only have a few plants, we report it with uncertainty: give or take.
9. **No coin.** Often we can't flip coins. We just have records. Then we compare like with like: watered vs. unwatered among sunny plants, and again among shady plants. This works only if we measured the things that matter. That's an assumption, and we have to say it out loud. And careful: don't adjust for things the watering itself changed, like leaf count after watering.
10. **ML.** Where does machine learning come in? Real gardens have many details: soil, pot size, species. ML handles lots of measured variables flexibly and can estimate how the effect differs for each kind of plant. But notice the tag is still on the machine. ML doesn't remove the assumption; it works inside it.
11. **Callback.** Back to our two plants. "What will grow?" is a prediction question: learn patterns, check accuracy. "What changes if we act?" is a causal question: it needs a fair comparison and stated assumptions. Even plain linear regression can serve either one. Better accuracy alone never proves that watering works.

## 7. Fact check

- Slide 5's sun -> watering and sun -> height links are **story assumptions**, stated in the speaker note ("in this garden").
- The treatment is *watering* (the assignment), not "the plant absorbed water".
- Slide 8: the effect is the **average** treatment effect, with sampling uncertainty. There are no invented numbers anywhere.
- Slide 9 contains the key cautions: no unmeasured common causes, and no adjusting for post-treatment variables.
- Slide 10: ML estimates effects that differ across groups and handles many covariates; it does not identify causality by itself.
- Slide 11: linear regression can support either task. Accuracy alone does not establish an intervention effect.

## 8. Production steps

1. Generate slide 1 with the base prompt + slide-1 block, attaching the existing "What will grow?" image as a style reference. Approve it: this is the anchor.
2. Generate slides 2-11 one at a time: base + continuity (with n) + slide block, attaching the anchor.
3. Fix any misspelled label with the local-edit prompt in `prompt-template.md`. Don't regenerate the whole image for one bad word.
4. Save as `01-hook.png` ... `11-callback.png` in this folder.
5. Put the images on a contact sheet and run series QA (`talk-mode.md` §7).
6. Build the deck: full-bleed image, the title from the beat sheet typeset in a clean sans-serif, and the notes above in the notes field.

**Shorter 6-minute cut:** slides 1, 2, 4, 5, 7, 11.
