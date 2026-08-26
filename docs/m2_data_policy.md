# M2 Data and Leakage Policy

HotpotQA split membership must be explicit at load time and carried on every
example. Training prompts may come only from the approved training split;
held-out evaluation examples must never contribute policy rewards or updates.

The first baseline uses the official distractor context attached to each
question as its local retrieval corpus. Search sees titles and passage text,
but never the gold answer field, supporting-fact labels, or sentence indices.
Those labels are retained only for offline evaluation and retrieval diagnosis.

Dataset files are immutable inputs. Record their source, filename, byte size,
and SHA-256 digest in the run metadata. Do not silently reshuffle examples;
sampling requires a recorded seed and stable example-ID ordering.

Before accepting M2, manually inspect trajectories for answer leakage and
verify that removing search observations changes the model context and behavior.

For prompt development, reserve validation indices 0–99 in the normalized,
stable dataset order. Freeze the baseline prompt before evaluating indices 100
and above. Prompt-development examples must not appear in held-out result
tables or project claims.
