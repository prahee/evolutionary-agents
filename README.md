# Because Mom Said So: Priors in Evolutionary Agents
**COSC 89.34/189 Final Project** by {promita.rahee.sikder,ali.azam}.28@dartmouth.edu

## Usage
Make sure to set up your Dartmouth Chat API key and LangChain (via developer.dartmouth.edu) in the .env file. There is an example .env file present for refernece.

Then run
```bash
uv sync
uv run_experiments --exp all --trials 50 # Viability study
uv run_experiments --exp a --trials 250  # Main experiment on testing priors
```

There are some other code artifacts from failed experiments and future work, such as RGGs using cloaking as in graph theory research. Results should be generated in the `results/` directory.
