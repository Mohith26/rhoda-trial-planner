# Trial planner for real-robot evaluations (concept demo for Rhoda AI)

**Live demo:** https://mohith26.github.io/rhoda-trial-planner/

Rhoda's post *Does Scaling Web-Video Pre-training Help Real Robots Do Real Work?* (Sep 10, 2026) took over 200 robot-hours and 852 trials on seven checkpoints for its headline figure, and reports results the TRI way: Beta posterior violins with Fisher + Holm letters. This is a concept for the tool that sits next to that protocol and answers three planning questions with the trials already spent. Everything runs on the **published counts and DINO FD values**; no Rhoda code or data beyond the post is used.

1. **Is the trend real?** Pairwise letters test whether two checkpoints differ. A scaling claim is a trend claim. On the same 611 compute-sweep trials where only 2 of 6 pairs separate, a binomial logistic trend in log compute gives **p ≈ 0.0013** (+4.5 pp per doubling), and an order-restricted (isotonic) likelihood-ratio test gives p ≈ 0.0005 (bootstrap floor). The isotonic fit is flat from 0.37× to 1×, matching the post's saturation reading.
2. **Forecast a checkpoint before it touches the robot.** A logistic fit of at-speed completion on −log DINO FD across the seven checkpoints (with a quasi-binomial dispersion φ ≈ 2.3, because checkpoints scatter more than trial noise) turns a new checkpoint's FD into a forecast interval and into **trials per arm needed to show it beats L or M·1×** with 80% power, averaged over the forecast uncertainty.
3. **Spend the next trials.** Simulated power of the trend test under different allocations at the observed rates: putting 300 trials at the two ends of the sweep matches the published 611-trial design's power (94%); ends plus a middle point keeps the saturation check at 91% with 300.
4. **Looking early.** If Fisher + Holm letters are recomputed after every 10-trial session, the family-wise false-separation rate rises from 5% to about 14% (simulation with four identical policies). Anytime-valid e-values (Beta-mixture likelihood ratio against the pooled null, Holm across pairs) stay under 5% at every look, at a measurable cost in power (on the model-size counts they separate 3 of 6 pairs where Fisher separates 4).

## Validation against the post

`node --test test/evalplan.test.mjs` (11 tests) checks that the code reproduces the post's own numbers before computing anything new: the Beta(k+1, n−k+1) 95% credible intervals (e.g. 69.7%–80.1% for M·1×), the Fisher + Holm letters of Figure 3 (a, b, ab) and Figure 4 (a, b, bc, c), and the post's worked Fisher examples (75% vs 65% on 100 trials, p ≈ 0.16; 75% vs 60%, p ≈ 0.03). It also checks the logistic fit against a closed form, PAVA, the level of the trend test under a flat truth, the e-process's validity under peeking, the allocation claim, and forecast monotonicity. `node scripts/report.mjs` prints every headline number.

## Honest limitations

- **Seven checkpoints, one architecture family, one run each.** The FD-to-performance curve is a correlation across seven points (as the post says); the forecast's interval is widened for overdispersion but cannot cover a change in recipe.
- **Simulated power** uses the observed rates as the truth, so it inherits their sampling error.
- **Two-point designs** maximize power for a monotone trend but cannot see curvature; the page shows the three-point alternative for that reason.
- **Trend tests answer a different question** than pairwise letters; they do not say which checkpoint is best.
- **Anytime-valid letters cost power** at a fixed trial count; they are the right tool only when early stopping or repeated looks are actually in play.
- No in-app screenshots of Rhoda's internal tools are public, so the UI follows rhoda.ai's public design system (research-post layout, chart CSS, Okabe-Ito palette, Gotham/Sentinel stacks with their Montserrat/Bitter fallbacks).

## Novelty note

Rhoda's posts use Beta posteriors, Fisher + Holm letters and (for one interaction) a logistic regression; I found no trend test, FD-based trial forecasting, allocation planning or anytime-valid letters in Rhoda's public material. In the literature, STEP (Snyder et al., 2024) and SureSim (Badithela et al., 2025) address sequential policy comparison and sim-plus-real evaluation; this demo applies trend tests, optimal allocation and a pre-training-quality forecast to Rhoda's own evaluation protocol and published counts.

---
Unofficial concept demo by Mohith Gajjela. Not affiliated with Rhoda AI. Public data from rhoda.ai; simulations seeded and reproducible.
