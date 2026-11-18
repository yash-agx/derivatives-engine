# Phase 7 pre-registration: Longstaff-Schwartz against the binomial tree

Written and committed **before any Longstaff-Schwartz (LSM) price was computed**. The reference
values below come from the tree only. Nothing here may be changed after the results are in; if
a choice turns out to be wrong, the README says so and this file stays as it was.

## What is being tested

LSM prices an American put by simulation: fit the exercise rule by regressing realised
discounted cash flows on a basis of the stock price, using in-the-money paths only, then price
with that rule. The question: **does the out-of-sample LSM price agree with the tree, within its
own 95% confidence interval, on the twenty cases of Longstaff & Schwartz (2001), Table 1?**

## The test cases

American put, `K = 40`, `r = 0.06`, no dividends, every combination of

- `S` = 36, 38, 40, 42, 44
- `sigma` = 0.20, 0.40
- `T` = 1, 2 years

20 cases, numbered 0-19 in the order of the table below (S outermost, then sigma, then T).

## LSM settings (fixed for the validation run)

| setting | value |
|---|---|
| exercise dates | 50 per year (50 for T=1, 100 for T=2), as in the paper |
| basis | Laguerre polynomials `L_0 ... L_3` of `x = S/K` (degree 3, unweighted: 4 columns) |
| regression | in-the-money paths only, `numpy.linalg.lstsq` |
| exercise rule | exercise when the immediate payoff is strictly greater than the fitted continuation value |
| training paths | 100,000 |
| pricing paths | 100,000, independent of the training paths |
| seed | case `i` uses seed `7000 + i`; training and pricing streams come from `SeedSequence(seed).spawn(2)` |
| quoted estimate | **out-of-sample** (rule fitted on training paths, priced on pricing paths) |
| also reported | in-sample estimate (priced on the training paths) and the gap between the two |

## Reference values

**Primary reference:** the existing American CRR tree (`american_binomial`) at 10,000 steps.
Its uncertainty is estimated as `|V(10,000) - V(20,000)|`.

**Secondary reference:** a Bermudan tree (`exercise_tree.bermudan_binomial`) exercisable only on
the same 50-per-year dates as LSM (and at t = 0), with 10,000 steps in total. LSM prices this
contract, not the American one, so the secondary reference separates LSM's own error from the cost
of exercising on discrete dates. Its uncertainty is the change when its steps are doubled.

| case | S | sigma | T | American (10k) | tree uncert. | Bermudan, 50/yr | tree uncert. | European (BS) |
|---|---|---|---|---|---|---|---|---|
| 0 | 36 | 0.2 | 1 | 4.4867 | 1.2e-05 | 4.4779 | 3.6e-05 | 3.8443 |
| 1 | 36 | 0.2 | 2 | 4.8483 | 4.1e-06 | 4.8403 | 2.5e-05 | 3.7630 |
| 2 | 36 | 0.4 | 1 | 7.1090 | 1.2e-05 | 7.1013 | 9.0e-06 | 6.7114 |
| 3 | 36 | 0.4 | 2 | 8.5143 | 8.5e-05 | 8.5069 | 1.1e-04 | 7.7000 |
| 4 | 38 | 0.2 | 1 | 3.2572 | 6.8e-06 | 3.2502 | 1.7e-05 | 2.8519 |
| 5 | 38 | 0.2 | 2 | 3.7514 | 2.2e-05 | 3.7448 | 8.6e-06 | 2.9906 |
| 6 | 38 | 0.4 | 1 | 6.1547 | 1.4e-04 | 6.1477 | 1.5e-04 | 5.8343 |
| 7 | 38 | 0.4 | 2 | 7.6750 | 7.3e-05 | 7.6682 | 9.7e-05 | 6.9788 |
| 8 | 40 | 0.2 | 1 | 2.3195 | 1.4e-05 | 2.3141 | 4.9e-06 | 2.0664 |
| 9 | 40 | 0.2 | 2 | 2.8899 | 2.1e-05 | 2.8846 | 2.1e-06 | 2.3559 |
| 10 | 40 | 0.4 | 1 | 5.3182 | 3.7e-05 | 5.3119 | 4.9e-05 | 5.0596 |
| 11 | 40 | 0.4 | 2 | 6.9234 | 4.7e-05 | 6.9170 | 2.5e-05 | 6.3260 |
| 12 | 42 | 0.2 | 1 | 1.6212 | 4.4e-06 | 1.6170 | 5.5e-06 | 1.4645 |
| 13 | 42 | 0.2 | 2 | 2.2168 | 3.7e-05 | 2.2124 | 5.5e-05 | 1.8414 |
| 14 | 42 | 0.4 | 1 | 4.5882 | 7.1e-05 | 4.5825 | 5.9e-05 | 4.3787 |
| 15 | 42 | 0.4 | 2 | 6.2504 | 1.2e-04 | 6.2445 | 1.4e-04 | 5.7356 |
| 16 | 44 | 0.2 | 1 | 1.1130 | 2.7e-06 | 1.1099 | 3.9e-06 | 1.0169 |
| 17 | 44 | 0.2 | 2 | 1.6933 | 1.0e-06 | 1.6899 | 1.3e-05 | 1.4292 |
| 18 | 44 | 0.4 | 1 | 3.9528 | 2.7e-05 | 3.9477 | 3.3e-05 | 3.7828 |
| 19 | 44 | 0.4 | 2 | 5.6469 | 1.6e-04 | 5.6414 | 1.9e-04 | 5.2020 |

The tree is usable as a reference only if its uncertainty is far below LSM's. **Reference check:**
every tree uncertainty must be below 0.001. It is (largest 1.9e-04), so this check passes before
any LSM run. The American-minus-Bermudan gap is 0.003-0.009, which is the most that more paths
could never remove at 50 dates a year.

## What counts as a pass

1. **Per case (primary):** the American reference lies inside the 95% confidence interval of the
   out-of-sample LSM estimate: `|LSM - American| <= 1.96 * SE`.
2. **Overall (primary):** at least **18 of 20** cases pass. With 20 independent tests at 95% and an
   unbiased estimator, three or more misses happens 7.5% of the time by chance alone, so 3+
   misses is treated as evidence of a real problem rather than bad luck.
3. **Secondary:** the same two rules against the Bermudan reference.
4. **Direction of misses:** the out-of-sample estimate is low-biased for the Bermudan price, which
   is itself below the American. Misses are therefore expected, if at all, *below* the reference.
   A miss above the reference would contradict the theory and is reported separately.
5. The in-sample estimate is **not** a pass criterion. It is reported with its gap to the
   out-of-sample estimate.

If the overall rule fails, the README reports LSM as failing this validation at this budget,
and gives the Bermudan comparison as the diagnosis. It does not re-run with different seeds or
settings and quote the better run.

## Predictions for the convergence experiments (run on case 0 unless stated)

Written down so they can be checked, not because they are certain:

- **Paths:** RMSE against the Bermudan reference falls roughly like `1/sqrt(N)`. Against the
  American reference it flattens at about the Bermudan gap (0.009 for case 0).
- **In-sample vs out-of-sample:** the in-sample mean is above the out-of-sample mean on average
  over seeds, and the gap shrinks as the number of training paths grows.
- **Exercise dates:** the Bermudan value rises monotonically towards the American value as dates
  increase; one date gives the European price (or the immediate payoff, if larger); LSM follows
  the Bermudan tree, not the American one.
- **Basis degree:** degree 0 (a constant) is clearly worse (lower out-of-sample price). Degrees 2
  and 3 are within noise of each other; higher degrees add little.
- **Exercise boundary:** the LSM boundary lies close to the tree's, and is noisiest close to expiry
  and where few paths are in the money.
