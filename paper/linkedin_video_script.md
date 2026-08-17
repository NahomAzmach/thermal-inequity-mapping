# LinkedIn Video Script — "I said the heat wasn't there. Then I looked at night."

**Target length:** ~85–100 seconds (≈210–240 words spoken)
**Tone:** curious, honest, a real "wait, no—" turn. Not hype.
**Visuals you have:** `figures/panels_2024.png` (RGB | class | LST), `figures/change_map.png`, `figures/lst_by_class_year.png`, `figures/day_vs_night_effect.png` (the money shot), `figures/core_vs_fringe_night.png`.

---

## The script (voiceover + on-screen)

**[0:00–0:09] HOOK**
🎙️ "I used a Google AI satellite model to test whether my home city's slums are heat traps. The daytime data said no. I almost stopped there — that would've been a mistake."
🖥️ *RGB of Addis zooming in. Text: "Are informal settlements heat traps?"*

**[0:09–0:26] THE SETUP**
🎙️ "Every study of Addis Ababa's urban heat uses hand-crafted formulas. I tried something new: Google's AlphaEarth foundation model turns every 10-metre patch of Earth into a 64-number fingerprint. I trained a simple classifier on it to separate informal settlements from the formal city."
🖥️ *RGB wipes into the red/grey class map (`panels_2024.png`).*

**[0:26–0:38] IT WORKS**
🎙️ "About 70 hand-drawn examples, 94% accuracy — and running it on 2017 and 2024, I could watch the settlements expand into the city's edges."
🖥️ *`change_map.png`, red expansion pixels animate in.*

**[0:38–0:50] TWIST #1**
🎙️ "Then the temperature test. I expected informal areas to be hotter. They came out slightly *cooler* — even after controlling for elevation and greenery. Hypothesis: dead."
🖥️ *`lst_by_class_year.png`, arrow to informal (red) sitting lower.*

**[0:50–1:08] TWIST #2 — the turn**
🎙️ "But satellites take that picture at 10:30 in the morning. Heat hurts people at *night*. So I pulled nighttime thermal data — and the whole thing flipped. After dark, informal fabric runs up to a degree *hotter*. The penalty was real. Daytime just couldn't see it."
🖥️ *`day_vs_night_effect.png` — orange 'day' bars below zero, blue 'night' bar jumping above. Big text: "Day: cooler. Night: HOTTER."*

**[1:08–1:22] THE NUANCE (credibility)**
🎙️ "And it's the old, dense core that traps heat overnight — not the newly-built edges, which haven't packed in yet. The city is literally growing into its own heat problem."
🖥️ *`core_vs_fringe_night.png` — tall dark-red 'core' bar vs flat 'fringe' bar.*

**[1:22–1:32] CLOSE**
🎙️ "The lesson: whether you *see* climate injustice depends on when you measure it. Ask the wrong question, get the wrong answer. Measure at night."
🖥️ *Back to class map. Text: "Right question > fancy model." + your name/handle.*

---

## Caption for the post

I used Google's AlphaEarth satellite foundation model to map Addis Ababa's informal settlements — no hand-crafted spectral indices, just learned embeddings + ~70 labels + a linear probe. 94% cross-validated AUC, and a clear map of settlement expansion from 2017→2024.

Then a twist, then a twist on the twist.

Daytime satellite temperature said informal areas were slightly *cooler* than the formal city — even after controlling for elevation and vegetation. Hypothesis apparently dead.

But daytime surface temperature is measured mid-morning, and urban heat hurts people at night. So I brought in nighttime thermal data — and the sign flipped. After dark, informal fabric runs up to ~1°C *hotter*. The penalty was real all along; the daytime snapshot just couldn't see it. And it concentrates in the established dense core, not the newly-expanded fringe — the city is growing into its heat problem.

The lesson I'm taking away: whether you detect climate injustice depends on *when* you measure. The fancy model mapped the city; asking the right question is what found the finding.

Next: nighttime air-temperature data, and pushing the labels from 70 toward 150.

#RemoteSensing #GeospatialAI #UrbanHeat #MachineLearning #Ethiopia #ClimateEquity #EarthEngine #FoundationModels

---

## Production notes
- **The money shot is `day_vs_night_effect.png`** at [0:50]. Time your voice so "flipped" lands exactly as the blue night bar pops above zero. That's the moment people share.
- Keep on-screen numbers big and few: **94% AUC**, **2017→2024**, **Day: −0.4°C / Night: +1°C**.
- The double honesty ("hypothesis dead" → "wait, check the night") is the whole engine. Don't rush [0:38–0:50]; let twist #1 land as a real dead end before you rescue it.
- Under 60s? Cut the setup [0:09–0:26] to one line and drop the nuance beat [1:08–1:22]; keep both twists.
