
## Results
Cross-validation accuracy: ~60-65% on this small (40-sentence) dataset.

## Known Limitations
- Small training vocabulary means unfamiliar words/phrasing (e.g. indirect complaints without obviously negative words) can be misclassified
- A single train/test split initially gave a misleadingly unstable accuracy score (0-25%) due to the tiny dataset; switched to cross-validation to fix this
- Would need a much larger, more diverse dataset (hundreds+ of examples) for production-level accuracy

## Future Improvements
- Train on a real public dataset (e.g. IMDB reviews) for significantly better accuracy
- Add a neutral category, not just positive/negative
- Try a different model (e.g. logistic regression) and compare performance

## What I learned
That small ML datasets create genuinely unreliable evaluation (a single split can show 0% accuracy even when the model works), how cross-validation solves that, and that vocabulary coverage — not just model choice — is often the real bottleneck in a text classifier's performance.
